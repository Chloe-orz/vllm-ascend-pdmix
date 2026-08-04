#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Inject ascend PD-separation / edge-cloud / passive-PP hooks into the
upstream :class:`vllm.v1.engine.core.EngineCore` and
:class:`vllm.v1.engine.core.EngineCoreProc` without modifying upstream
sources.

This patch is the home of every line that the vllm-pdmix downstream fork
used to maintain inside ``vllm/v1/engine/core.py`` of vLLM:

* ``EngineCore.__init__`` — late-stage construction of the optional
  edge-cloud PD-separation channel (``self._pp_pd_channel``).
* ``EngineCore.step`` / ``EngineCore.step_with_batch_queue`` — drain
  cloud-returned batches into the local PD scheduler, publish
  head-segment batches on PRE_OUT, skip ``sample_tokens`` for head
  batches, and assign ``head_token`` ids.
* ``EngineCore._drain_pd_channel_inbox`` /
  ``EngineCore._maybe_publish_pre_out`` /
  ``EngineCore._needs_sample_tokens`` — three new helper methods used by
  the two ``step*`` paths above.
* ``EngineCore.shutdown`` — release the channel before the rest of the
  engine resources.
* ``EngineCoreProc._process_input_queue`` — force a blocking
  ``input_queue.get`` when the engine has nothing local to do, so the
  edge node never busy-spins while waiting for the next client request.

Design notes
------------
1. ``__init__`` and ``shutdown`` only append behavior at the end and at
   the start, respectively, so they are wrapped (call original + extra).
2. ``step`` / ``step_with_batch_queue`` / ``_process_input_queue`` insert
   logic in the middle of the original method body. They are rewritten
   in full here, bytewise-equivalent to upstream when no PD/edge-cloud
   feature flag is on.
3. Every flag read uses ``getattr(parallel_config, ..., default)`` so
   that even if the dest-only ``ParallelConfig`` extension fields are
   absent, this patch behaves identically to upstream.
4. The patch is installed at import time. A guard prevents double
   patching if this module is imported twice (e.g. from a child
   process).

Upstream sync
-------------
The reimplementations of ``step()``, ``step_with_batch_queue()`` and
``_process_input_queue()`` track upstream
``vllm-0.20.2_layerwise/vllm/v1/engine/core.py``. Whenever vLLM moves to
a new minor version, re-diff these methods against the new upstream
source and re-apply the dest-only inserts.
"""
from __future__ import annotations

import functools
from concurrent.futures import Future
from typing import cast
from uuid import uuid4

from vllm.config import ParallelConfig
from vllm.logger import init_logger, logger as vllm_logger
from vllm.v1.core.sched.output import BatchType, SchedulerOutput
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from vllm_ascend.v1.engine.passive_core import PPSchedulerZmqChannel

logger = init_logger(__name__)


# Idempotency guard: re-importing this module (e.g. from a child process)
# must not double-wrap the original methods.
_INSTALLED_FLAG = "_vllm_ascend_engine_core_patched"


# -----------------------------------------------------------------------#
# Original method handles captured before any wrapping happens.           #
# -----------------------------------------------------------------------#
_ORIG_ENGINE_CORE_INIT = EngineCore.__init__
_ORIG_ENGINE_CORE_SHUTDOWN = EngineCore.shutdown
_ORIG_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core


# =======================================================================#
# EngineCore.__init__ — append PD/edge-cloud setup at the very end.       #
# =======================================================================#
@functools.wraps(_ORIG_ENGINE_CORE_INIT)
def _patched_engine_core_init(self, *args, **kwargs):
    _ORIG_ENGINE_CORE_INIT(self, *args, **kwargs)

    parallel_config: ParallelConfig = self.vllm_config.parallel_config

    # PD-separation is owned by the ascend plugin and lives under
    # ``additional_config.edge_cloud_config.pd_separation``. ``init_ascend_config``
    # is idempotent and returns the cached singleton if already initialized
    # in the main process; in a freshly-spawned subprocess it re-initializes
    # from the ``vllm_config`` we hold.
    from vllm_ascend.ascend_config import init_ascend_config
    ascend_config = init_ascend_config(self.vllm_config)
    edge_cloud = getattr(ascend_config, "edge_cloud_config", None)
    pd_enabled = bool(
        edge_cloud is not None
        and getattr(edge_cloud, "enabled", False)
        and getattr(edge_cloud, "pd_separation", None) is not None
        and edge_cloud.pd_separation.enabled
    )

    if getattr(parallel_config, "enable_edge_cloud", False):
        logger.info(
            "Edge-cloud mode enabled (pd_separation=%s)",
            pd_enabled,
        )

    # Load PD-separation configuration from environment variables
    from vllm_ascend.pd_separation_config import PDSeparationConfig
    pd_config = PDSeparationConfig.from_env()

    self.step_cnt = 0
    # Edge-cloud PD-separation bidirectional ZMQ channel (edge side).
    self._pp_pd_channel = None
    if pd_enabled and getattr(parallel_config, "is_edge_node", False):
        dp_rank = getattr(parallel_config, "data_parallel_rank", 0)
        import torch.distributed as dist
        from datetime import timedelta

        # Cloud cross-DP coord group: edge DP0 hosts a tiny IP-exchange
        # store so cloud DP1+ can discover cloud DP0's reachable IP (cloud
        # DP0 is the gloo rank-0 / store master). Hosted with
        # wait_for_workers=False so the constructor returns immediately
        # (no barrier, no blocking on cloud startup); kept alive on `self`
        # for the process lifetime. Clouds set/get ``coord_master_ip``
        # during their coord-group init. See passive_core.py
        # run_passive_engine_core for the client side.
        _dp_size = getattr(parallel_config, "data_parallel_size", 1)
        _is_moe = bool(
            getattr(self.vllm_config.model_config, "is_moe", False)
        )
        if dp_rank == 0 and _dp_size > 1 and _is_moe:
            self._cloud_coord_ip_store = dist.TCPStore(
                host_name=parallel_config.master_addr,
                port=parallel_config.master_port + 200,
                world_size=_dp_size,
                is_master=True,
                wait_for_workers=False,
                timeout=timedelta(seconds=300),
            )
            logger.info(
                "Edge DP0 hosting cloud-coord IP-exchange store on "
                "%s:%s",
                parallel_config.master_addr,
                parallel_config.master_port + 200,
            )

        # Discover the cloud's IP via a one-shot TCPStore. The edge
        # acts as store master on ``master_port + 1 + dp_rank`` so
        # that each DP rank has its own store port (no EADDRINUSE).
        # The cloud connects once per edge DP rank and writes its
        # ``get_ip()`` result. See passive_core.py for the
        # symmetric writer side.
        _addr_store = dist.TCPStore(
            host_name=parallel_config.master_addr,
            port=parallel_config.master_port + 1 + dp_rank,
            world_size=2,
            is_master=True,
            timeout=timedelta(seconds=300),
        )
        cloud_addr = _addr_store.get("cloud_ip").decode()
        del _addr_store

        # Each DP rank needs its own ZMQ port pair to avoid bind
        # conflicts within the same edge process. Offset by 2 per
        # dp_rank: dp_rank 0 → {pre_out, post_out}, dp_rank 1 →
        # {pre_out+2, post_out+2}, etc. The cloud side must mirror
        # this offsetting in its own PPSchedulerZmqChannel setup.
        pre_out_port = pd_config.pre_out_port + dp_rank * 2
        post_out_port = pd_config.post_out_port + dp_rank * 2
        pre_out = f"tcp://*:{pre_out_port}"
        post_out = f"tcp://{cloud_addr}:{post_out_port}"
        self._pp_pd_channel = PPSchedulerZmqChannel(
            send_endpoint=pre_out,
            recv_endpoint=post_out,
            name=f"pd-edge-dp{dp_rank}",
        )
        logger.info(
            "PD-separation edge channel: PRE_OUT=%s, POST_OUT=%s "
            "(cloud_addr=%s auto-discovered)",
            pre_out, post_out, cloud_addr,
        )


# =======================================================================#
# Three helper methods bound on EngineCore. Mirror the dest fork.         #
# =======================================================================#
def _drain_pd_channel_inbox(self) -> None:
    """Move cloud-returned SchedulerOutputs into the local PDSeparated
    scheduler's ``prefills_last_ready`` / ``decodes_last_ready`` queues.
    """
    if getattr(self, "_pp_pd_channel", None) is None:
        return
    if not (
        hasattr(self.scheduler, "prefills_last_ready")
        and hasattr(self.scheduler, "decodes_last_ready")
        and hasattr(self.scheduler, "drafts_last_ready")
    ):
        return
    new_outputs = self._pp_pd_channel.consume_new_outputs()
    for _seq, so in new_outputs:
        bt = so.batch_type
        logger.info(f"Received scheduler_output from cloud, batch_type: {bt}")
        if bt == BatchType.PREFILL_LAST:
            self.scheduler.prefills_last_ready.append(so)
        elif bt == BatchType.DECODE_LAST:
            self.scheduler.decodes_last_ready.append(so)
        elif bt == BatchType.DRAFT_LAST:
            # DRAFT_LAST is self-posted by _pick_draft_first_batch (like
            # DECODE_LAST). If it arrives via POST_OUT (e.g. from an older
            # cloud that still publishes it), drop it -- the edge already has
            # its own copy in drafts_last_ready.
            logger.debug(
                "Dropping POST_OUT DRAFT_LAST head_token=%s "
                "(edge self-posts DRAFT_LAST)",
                getattr(so, "head_token", None),
            )
        else:
            logger.error(
                "PD-separation POST_OUT received unexpected batch_type=%s; "
                "expected PREFILL_LAST, DECODE_LAST, or DRAFT_LAST. "
                "Dropping.",
                bt.value if bt is not None else "<none>",
            )


def _maybe_publish_pre_out(
    self, scheduler_output: SchedulerOutput
) -> None:
    """Forward head-segment batches on the edge → cloud channel."""
    if getattr(self, "_pp_pd_channel", None) is None:
        return
    bt = scheduler_output.batch_type
    if bt == BatchType.DRAFT_FIRST:
        is_pregenerated = getattr(
            self.scheduler, "is_pre_generated_draft", lambda _so: False
        )(scheduler_output)
        if is_pregenerated:
            task_id = scheduler_output.draft_task_id
            assert task_id is not None
            opened = getattr(
                self, "_pd_draft_pre_out_open_tasks", None
            )
            if opened is None:
                opened = set()
                self._pd_draft_pre_out_open_tasks = opened
            if task_id not in opened:
                # Edge dispatch is intentionally independent of cloud
                # readiness. Queue every cloud control in task order so later
                # placeholder steps cannot overtake step 0 while its
                # accepted-token scalars are still being finalized.
                deferred = getattr(
                    self, "_pd_deferred_draft_pre_out", None
                )
                if deferred is None:
                    deferred = {}
                    self._pd_deferred_draft_pre_out = deferred
                deferred.setdefault(task_id, []).append(scheduler_output)
                return
        self._pp_pd_channel.publish(scheduler_output)
    elif bt in (
        BatchType.PREFILL_FIRST,
        BatchType.DECODE_FIRST,
    ):
        self._pp_pd_channel.publish(scheduler_output)
    elif bt in (
        BatchType.EMPTY,
        BatchType.PREFILL_LAST,
        BatchType.DECODE_LAST,
        BatchType.DRAFT_LAST,
    ):
        return
    else:
        logger.debug(
            "PD-separation PRE_OUT skipping non-separated batch_type=%s",
            bt.value if bt is not None else "<none>",
        )


def _release_deferred_draft_pre_out(
    self, draft_task_id: str
) -> None:
    """Open one cloud draft control stream and flush it in FIFO order."""
    opened = getattr(self, "_pd_draft_pre_out_open_tasks", None)
    if opened is None:
        opened = set()
        self._pd_draft_pre_out_open_tasks = opened
    opened.add(draft_task_id)

    deferred = getattr(self, "_pd_deferred_draft_pre_out", None)
    queued = [] if deferred is None else deferred.pop(draft_task_id, [])
    channel = getattr(self, "_pp_pd_channel", None)
    if channel is None:
        return
    for scheduler_output in queued:
        channel.publish(scheduler_output)
    if queued:
        logger.info(
            "[PRE_OUT] released %d async draft controls task_id=%s",
            len(queued),
            draft_task_id,
        )


def _close_draft_pre_out(self, draft_task_id: str | None) -> None:
    if not draft_task_id:
        return
    opened = getattr(self, "_pd_draft_pre_out_open_tasks", None)
    if opened is not None:
        opened.discard(draft_task_id)
    deferred = getattr(self, "_pd_deferred_draft_pre_out", None)
    if deferred is not None:
        deferred.pop(draft_task_id, None)


def _ensure_pd_head_token(self, scheduler_output: SchedulerOutput) -> None:
    if getattr(self, "_pp_pd_channel", None) is None:
        return
    if scheduler_output.batch_type not in (
        BatchType.PREFILL_FIRST,
        BatchType.DECODE_FIRST,
        BatchType.DRAFT_FIRST,
    ):
        return
    if not scheduler_output.head_token:
        scheduler_output.head_token = uuid4().hex


def _ensure_pd_original_seq(self, scheduler_output: SchedulerOutput) -> None:
    """Stamp the edge-side original_seq on an EngineCore-created dummy.

    The scheduler stamps real head-segment batches (PF/DF/DRAFT_FIRST) and
    its own coordinated dummies at production time. Dummies built directly
    in the EngineCore for cross-DP coordination (see
    ``_patched_execute_dummy_batch`` / ``_publish_pd_dummy_zmq``) bypass
    that path, so stamp them here via the scheduler's counter. This keeps
    every edge-produced SchedulerOutput - including all DP-parallel dummies -
    on one monotonic original_seq sequence, so the cloud worker's [EC-EXEC]
    log never shows a dummy with original_seq=None.
    """
    _assign = getattr(self.scheduler, "_assign_original_seq", None)
    if _assign is not None:
        _assign(scheduler_output)


def _needs_sample_tokens(self, scheduler_output: SchedulerOutput) -> bool:
    """Return True if sample_tokens should follow execute_model for this
    batch.

    In edge-cloud PD-separation mode, only tail-segment batches (PL/DL)
    produce logits and need sampling. Head-segment batches (PF/DF) output
    intermediate hidden states and must skip sampling.
    """
    if getattr(self, "_pp_pd_channel", None) is None:
        return True
    bt = scheduler_output.batch_type
    return bt in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST)


def _stash_empty_worker_cleanup(self, scheduler_output: SchedulerOutput) -> None:
    """Keep worker-side cleanup from EMPTY batches for the next real batch."""
    finished_req_ids = getattr(scheduler_output, "finished_req_ids", None)
    free_encoder_mm_hashes = getattr(scheduler_output, "free_encoder_mm_hashes", None)
    if not finished_req_ids and not free_encoder_mm_hashes:
        return

    pending_finished = getattr(self, "_pd_pending_finished_req_ids", None)
    if pending_finished is None:
        pending_finished = set()
        self._pd_pending_finished_req_ids = pending_finished
    pending_finished.update(finished_req_ids or ())

    pending_mm_hashes = getattr(self, "_pd_pending_free_encoder_mm_hashes", None)
    if pending_mm_hashes is None:
        pending_mm_hashes = set()
        self._pd_pending_free_encoder_mm_hashes = pending_mm_hashes
    pending_mm_hashes.update(free_encoder_mm_hashes or ())


def _merge_pending_worker_cleanup(self, scheduler_output: SchedulerOutput) -> None:
    """Attach cleanup skipped with EMPTY batches to the next worker batch."""
    pending_finished = getattr(self, "_pd_pending_finished_req_ids", None)
    if pending_finished:
        scheduler_output.finished_req_ids = set(
            scheduler_output.finished_req_ids
        ).union(pending_finished)
        pending_finished.clear()

    pending_mm_hashes = getattr(self, "_pd_pending_free_encoder_mm_hashes", None)
    if pending_mm_hashes:
        scheduler_output.free_encoder_mm_hashes = list(
            dict.fromkeys([
                *scheduler_output.free_encoder_mm_hashes,
                *pending_mm_hashes,
            ])
        )
        pending_mm_hashes.clear()


def _finish_empty_batch(self, scheduler_output: SchedulerOutput):
    """Complete an EMPTY SchedulerOutput without broadcasting to workers."""
    self._stash_empty_worker_cleanup(scheduler_output)
    self._process_aborts_queue()
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT
        )
    self._clear_pending_edge_cloud_draft_for_finished_requests()
    return engine_core_outputs, False


def _defer_empty_batch(self, scheduler_output: SchedulerOutput) -> None:
    """Defer an EMPTY batch when queued model work must complete first."""
    deferred = getattr(self, "_pd_deferred_empty_batches", None)
    if deferred is None:
        deferred = []
        self._pd_deferred_empty_batches = deferred
    deferred.append(scheduler_output)


def _pop_deferred_empty_batch(self) -> SchedulerOutput | None:
    deferred = getattr(self, "_pd_deferred_empty_batches", None)
    if not deferred:
        return None
    return deferred.pop(0)


def _advance_edge_cloud_draft(
    self,
    completed_scheduler_output: SchedulerOutput,
    model_output: ModelRunnerOutput,
) -> None:
    """Advance scheduled draft control state after a completed tail batch.

    Draft SchedulerOutputs are generated by PDSeparatedScheduler, matching
    DECODE_FIRST/DECODE_LAST ownership. Cloud-only step-0 sampling state is
    derived from the already-returned target output. In async mode final draft
    token IDs remain worker-local and the scheduler advances with placeholders.
    """
    if not getattr(self, "use_spec_decode", False):
        return
    enqueue_draft_first = getattr(
        self.scheduler, "enqueue_draft_first", None
    )
    if enqueue_draft_first is None:
        return

    batch_type = completed_scheduler_output.batch_type
    # Every prefill chunk must run the drafter to populate the MTP KV cache.
    # Mid-chunk sampled/draft tokens are discarded by the worker, but their
    # independently scheduled draft chain still needs to be finalized here.
    is_target_tail = batch_type in (
        BatchType.PREFILL_LAST,
        BatchType.DECODE_LAST,
    )
    if is_target_tail:
        state = getattr(model_output, "edge_cloud_draft_state", None)
        if state is None:
            return
        task_id = state["draft_task_id"]
        num_accepted_tokens = state.get("num_accepted_tokens")
        valid_sampled_token_count = state.get(
            "valid_sampled_token_count"
        )
        if num_accepted_tokens is None:
            # AsyncModelRunnerOutput has already materialized and filtered
            # sampled_token_ids before EngineCore receives it. Their row
            # lengths are exactly the accepted counts, so reuse that existing
            # D2H result instead of synchronizing a second copy in the edge
            # worker. The same counts drive async batch-state correction.
            num_accepted_tokens = [
                len(token_ids)
                for token_ids in model_output.sampled_token_ids
            ]
            valid_sampled_token_count = list(num_accepted_tokens)
        finalize = getattr(
            self.scheduler, "finalize_pre_generated_draft_first", None
        )
        if (
            batch_type in (
                BatchType.PREFILL_LAST,
                BatchType.DECODE_LAST,
            )
            and finalize is not None
        ):
            finalized = finalize(
                draft_task_id=task_id,
                num_accepted_tokens=num_accepted_tokens,
                valid_sampled_token_count=valid_sampled_token_count,
            )
            if finalized is not None:
                # This only opens the cloud control stream. Edge DRF/DRL
                # SchedulerOutputs were already dispatched independently and
                # never wait for accepted-token propagation.
                self._release_deferred_draft_pre_out(task_id)
                return
        enqueue_draft_first(
            completed_scheduler_output,
            draft_task_id=task_id,
            draft_step_idx=int(state["draft_step_idx"]),
            num_accepted_tokens=num_accepted_tokens,
            valid_sampled_token_count=valid_sampled_token_count,
        )
        return

    if batch_type != BatchType.DRAFT_LAST:
        return
    draft_step_idx = int(completed_scheduler_output.draft_step_idx or 0)
    if draft_step_idx + 1 >= getattr(
        self.scheduler, "num_spec_tokens", 0
    ):
        self._close_draft_pre_out(
            completed_scheduler_output.draft_task_id
        )
        # The draft chain has fully executed on the cloud; release the KV
        # blocks retained for requests that finished while the chain was
        # in flight.
        release = getattr(
            self.scheduler, "release_draft_retained_blocks", None
        )
        task_id = completed_scheduler_output.draft_task_id
        if release is not None and task_id:
            release(task_id)
    draft_token_ids = getattr(
        model_output, "edge_cloud_draft_token_ids", None
    )
    if draft_token_ids is not None:
        self.scheduler.update_draft_token_ids(draft_token_ids)


def _clear_pending_edge_cloud_draft_for_finished_requests(self) -> None:
    """Push draft lifecycle events across the executor boundary.

    Two duties, both driven by scheduler-side state:
      1. Forward finished request ids to the edge worker so it can mark
         pending draft contexts.  A context is dropped there only when
         EVERY request of its parent batch has finished; partial finishes
         keep the chain — the cloud-side cached attention metadata is
         whole-batch and cannot be re-sliced.
      2. Drain draft task ids the scheduler cut from its ready queues:
         release their retained KV blocks, invalidate the cloud-side
         cached draft metadata, and force-drop any runner context still
         held for a cut chain.
    """
    if not getattr(self, "use_spec_decode", False):
        return
    finished_req_ids = set(
        getattr(self.scheduler, "finished_req_ids", set()) or ()
    )
    take_sched_dropped = getattr(
        self.scheduler, "take_dropped_draft_task_ids", None
    )
    sched_dropped = (
        take_sched_dropped() if take_sched_dropped is not None else []
    )
    release = getattr(
        self.scheduler, "release_draft_retained_blocks", None
    )
    if release is not None:
        for task_id in sched_dropped:
            release(task_id)
    invalidate = getattr(
        self.scheduler, "invalidate_cloud_draft_tasks", None
    )
    if invalidate is not None:
        invalidate(sched_dropped)
    if not finished_req_ids and not sched_dropped:
        return
    clear_pending = getattr(
        self.model_executor,
        "clear_pending_edge_cloud_draft_for_req_ids",
        None,
    )
    if clear_pending is not None:
        clear_pending(finished_req_ids, sched_dropped)


def _register_edge_cloud_draft_parent(
    self,
    scheduler_output: SchedulerOutput,
    model_output: ModelRunnerOutput,
) -> None:
    """Retain a parent request when its worker created a draft task."""
    if not getattr(self, "use_spec_decode", False):
        return
    if not self._uses_scheduled_edge_cloud_draft():
        return
    if scheduler_output.batch_type not in (
        BatchType.PREFILL_LAST,
        BatchType.DECODE_LAST,
    ):
        return
    # Pregeneration is only an optimization and can be skipped while another
    # draft chain occupies the local queue.  The worker state is authoritative:
    # _advance_edge_cloud_draft() will enqueue a fallback chain whenever this
    # state exists, so its parent KV must be retained even without a
    # pre-generated SchedulerOutput.
    state = getattr(model_output, "edge_cloud_draft_state", None)
    if state is None:
        return
    task_id = state.get("draft_task_id")
    parent_task_id = getattr(scheduler_output, "head_token", None)
    if task_id != parent_task_id:
        raise RuntimeError(
            "Edge-cloud draft parent task mismatch: "
            f"scheduler={parent_task_id}, worker={task_id}"
        )
    req_ids = set(scheduler_output.num_scheduled_tokens)
    register = getattr(
        self.scheduler, "register_edge_cloud_draft_task", None
    )
    if register is not None and task_id and req_ids:
        register(task_id, req_ids)


def _uses_scheduled_edge_cloud_draft(self) -> bool:
    speculative_config = self.vllm_config.speculative_config
    if (
        getattr(self, "_pp_pd_channel", None) is None
        or speculative_config is None
    ):
        return False
    method = getattr(speculative_config, "method", None)
    if method == "eagle3":
        return True
    if method in ("qwen3_5_mtp", "qwen_mtp"):
        return True
    if method != "mtp":
        return False
    hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
    return "qwen" in str(getattr(hf_config, "model_type", "")).lower()


def _has_unresolved_edge_cloud_draft_parent(self) -> bool:
    """Keep async scheduling behind a prefill tail only.

    Async scheduled-MTP pre-generates the decode draft chain when DECODE_LAST
    is picked, so DECODE_LAST must not hold back local edge dispatch.
    """
    if not self._uses_scheduled_edge_cloud_draft():
        return False
    batch_queue = getattr(self, "batch_queue", None)
    if not batch_queue:
        return False
    for _future, scheduler_output, _exec_future in batch_queue:
        if (
            scheduler_output.batch_type == BatchType.PREFILL_LAST
            and getattr(scheduler_output, "is_last_prefill_chunk", True)
        ):
            is_pregenerated = getattr(
                self.scheduler,
                "is_pre_generated_draft",
                lambda _so: False,
            )(scheduler_output)
            if not is_pregenerated:
                return True
    return False


# =======================================================================#
# EngineCore.step — full replacement, mirrors upstream + dest inserts.    #
# =======================================================================#
# batch_type id packed into the coordination all_reduce. 0 = EMPTY/idle,
# 1/2/3/4 = real PF/PL/DF/DL. A dummy SchedulerOutput carries the WINNER bt
# (NOT 0): both DPs contribute the same winner id, so peer_bt != 0 and the
# _dummy_run skip-both branch (peer_bt == 0) never fires under coordination
# - this is what kills the old self-driven-dummy count-drift deadlock path.
_BT_COORD_ID = {
    BatchType.EMPTY: 0,
    BatchType.PREFILL_FIRST: 1,
    BatchType.PREFILL_LAST: 2,
    BatchType.DECODE_FIRST: 3,
    BatchType.DECODE_LAST: 4,
}
_BT_COORD_ID_INV = {v: k for k, v in _BT_COORD_ID.items()}


def _is_coordinated_dp(self) -> bool:
    """True when edge-side cross-DP batch_type coordination is active:
    dp>1 (DPEngineCoreProc owns dp_group) + PD-separation channel + MoE."""
    return (
        getattr(self, "dp_group", None) is not None
        and getattr(self, "_pp_pd_channel", None) is not None
        and bool(getattr(self.vllm_config.model_config, "is_moe", False))
    )


def _coordinate_bt(
    self, intended_bt: BatchType, local_unfinished: bool,
    local_has_work_waiting: bool = False,
) -> tuple[BatchType, bool]:
    """All-reduce intended batch_type, has_unfinished, and a decode-waiting
    flag across DPs in ONE all_reduce on dp_group, every step. Returns
    (winner, engines_running).

    Layout (SUM-gather, same trick as model_runner
    _sync_metadata_across_dp):
      [bt_id_0 .. bt_id_{n-1},
       wait_0  .. wait_{n-1},     # 1 if DP r has real decode work but
                                  # intended EMPTY (waiting for its
                                  # DECODE_LAST to return from cloud)
       unfinished_local]          # SUM = count of running DPs
    Each rank fills its own bt_id and wait slots (others 0); every rank
    adds its 0/1 unfinished at the last slot.

    Default winner = lowest-dp_rank DP with non-EMPTY intent (dp0
    priority); all EMPTY -> EMPTY. engines_running = OR(local_unfinished)
    (sum > 0). Each DP then force-schedules the winner (real if it has
    such work, else a dummy of the winner bt).

    Decode phase alignment (ONLY when BOTH DPs are on the decode side,
    i.e. every intended in {EMPTY, DECODE_FIRST, DECODE_LAST}; prefill
    PF/PL is left to the default rule):

      Rule 2 - any DP has DECODE_LAST ready -> winner = DECODE_LAST.
        Lets the lagging DP finish its DL (real) while the DF-ready DP
        waits (dummy DL); next step both can DF together. Without this
        the DF-ready DP (usually dp0) grabs the step as DF and the peer's
        DL never runs -> phase stays skewed -> perpetual
        one-real-one-dummy.

      Rule 1 - one DP DF-ready, the other EMPTY *with waiting decode
        work* -> winner = EMPTY this step. Defers the DF so neither runs
        a dummy DF; once the waiting DP's DL arrives (-> Rule 2 -> real
        DL) both DPs reach DF together next step -> two real DF, dummy
        eliminated. If the EMPTY peer is truly idle (no waiting work),
        keep default DF so the dummy still pairs the cross-DP EP
        all-toall (necessary).

    Trade-off: alignment makes the leading DP wait (1-2 dummy/empty
    steps) for the lagging DP to catch up. Under balanced load this
    removes the steady-state one-real-one-dummy (edge dummy segment_a +
    cloud dummy full middle), ~2x decode throughput. Under cloud-return
    jitter or skewed load the leading DP may stall waiting, hurting
    latency - so Rule 1 only fires when the EMPTY peer actually has
    decode work waiting, never when idle.

    Reuses the EngineCore stateless dp_group (same communicator the
    worker uses for sync_metadata). Safe because coord mode runs this
    single all_reduce every step on both DPs (the busy loop gates
    `continue` on `not _is_coordinated_dp()`), so the per-group call
    count is always paired.
    """
    import torch
    dp_group = getattr(self, "dp_group", None)
    if dp_group is None:
        return intended_bt, bool(local_unfinished)
    parallel_config = self.vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    # Layout: [bt_id_0..bt_id_{n-1}, wait_0..wait_{n-1}, unfinished].
    # Each rank fills its own bt_id and wait index (others 0) so SUM
    # gathers per-rank values; every rank adds its 0/1 unfinished at the
    # last slot so SUM there = count of running DPs (>0 => engines_running).
    tensor = torch.zeros(2 * dp_size + 1, dtype=torch.int32, device="cpu")
    tensor[dp_rank] = _BT_COORD_ID.get(intended_bt, 0)
    tensor[dp_size + dp_rank] = 1 if local_has_work_waiting else 0
    tensor[2 * dp_size] = 1 if local_unfinished else 0
    _cnt = getattr(self, "_coord_bt_count", 0) + 1
    self._coord_bt_count = _cnt
    torch.distributed.all_reduce(tensor, group=dp_group)
    bt_ids = [int(tensor[r].item()) for r in range(dp_size)]
    waitings = [int(tensor[dp_size + r].item()) for r in range(dp_size)]
    _engines_running = int(tensor[2 * dp_size].item()) > 0

    # Default: lowest-dp_rank non-EMPTY intended (dp0 priority).
    winner_id = 0
    for r in range(dp_size):
        if bt_ids[r] != 0:
            winner_id = bt_ids[r]
            break

    _DL = _BT_COORD_ID[BatchType.DECODE_LAST]
    _DF = _BT_COORD_ID[BatchType.DECODE_FIRST]
    _E = _BT_COORD_ID[BatchType.EMPTY]
    # Decode phase alignment - only when no DP is on the prefill side
    # (PF/PL). Prefill phases keep the default lowest-rank rule.
    if all(b in (_E, _DF, _DL) for b in bt_ids):
        if _DL in bt_ids:
            # Rule 2: let the DP with a ready DL run it (real); the
            # DF-ready peer runs a dummy DL so both align on DF next step.
            winner_id = _DL
        elif _DF in bt_ids and _E in bt_ids:
            # Rule 1: defer DF when the EMPTY peer is waiting on its DL
            # (has decode work). If the EMPTY peer is truly idle, fall
            # through to the default DF so the dummy pairs the a2a.
            if any(bt_ids[r] == _E and waitings[r] for r in range(dp_size)):
                winner_id = _E

    _winner = _BT_COORD_ID_INV.get(winner_id, BatchType.EMPTY)
    # Throttle: log every 32 calls and whenever real work is coordinated.
    if _cnt % 32 == 0 or winner_id != 0:
        logger.info(
            "[COORD] dp_rank=%s count=%s intended=%s winner=%s "
            "engines_running=%s bt_ids=%s waitings=%s",
            dp_rank, _cnt,
            _BT_COORD_ID.get(intended_bt, 0), _BT_COORD_ID.get(_winner, 0),
            int(_engines_running), bt_ids, waitings,
        )
    return _winner, _engines_running


def _patched_step(self):
    """Schedule, execute, and make output.

    Returns tuple of outputs and a flag indicating whether the model
    was executed.
    """
    # Check for any requests remaining in the scheduler - unfinished,
    # or finished and not yet removed from the batch.
    if not self.scheduler.has_requests():
        return {}, False

    # [ascend insert] Drain POST_OUT (cloud → edge) into the
    # PDSeparatedScheduler's tail-segment ready queues before scheduling.
    self._drain_pd_channel_inbox()

    scheduler_output = self.scheduler.schedule()
    self._ensure_pd_head_token(scheduler_output)

    # [ascend insert] Forward head-segment batches on the PRE_OUT
    # (edge → cloud) channel.
    self._maybe_publish_pre_out(scheduler_output)

    if scheduler_output.batch_type == BatchType.EMPTY:
        return self._finish_empty_batch(scheduler_output)

    self._merge_pending_worker_cleanup(scheduler_output)

    future = self.model_executor.execute_model(
        scheduler_output, non_block=True
    )
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)

    # Before processing the model output, process any aborts that happened
    # during the model execution.
    # Register the deferred draft before abort/model completion can free
    # requests referenced by this parent batch.
    self._register_edge_cloud_draft_parent(scheduler_output, model_output)
    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    self._advance_edge_cloud_draft(scheduler_output, model_output)
    self._clear_pending_edge_cloud_draft_for_finished_requests()

    return (
        engine_core_outputs,
        scheduler_output.total_num_scheduled_tokens > 0,
    )


# =======================================================================#
# EngineCore.step_with_batch_queue — full replacement.                    #
# =======================================================================#
def _patched_step_with_batch_queue(self):
    """Continuously fill the local worker FIFO before collecting one result.

    Native async scheduling relies on queue order, not EngineCore seeing each
    token first.  Fill all currently derivable DF/DL/DRF/DRL controls up to
    the executor credit so short edge segments cannot drain the MQ between
    two EngineCore turns.
    """
    batch_queue = self.batch_queue
    assert batch_queue is not None

    assert len(batch_queue) < self.batch_queue_size


    # [ascend insert] DP-parallel config diagnostic: judge via a cached flag
    # (self._dp_parallel) whether DP parallel is configured
    # (data_parallel_size>1). Config is static, so compute the flag once and
    # gate the one-time log purely on the flag (+ _dp_parallel_logged).
    _dp_parallel = getattr(self, "_dp_parallel", None)
    if _dp_parallel is None:
        # DP-parallel is configured only when data_parallel_size > 1. For DP=1
        # leave _dp_parallel as None (do NOT cache the `> 1` bool) so the
        # `if _dp_parallel is None:` branch below -- the simple, draft-guarded
        # path that yields correct MTP output -- is taken. Caching a bool here
        # makes `is None` always False and dead-codes that branch (878918df
        # regression: MTP draft race -> 不说人话).
        if getattr(
            self.vllm_config.parallel_config, "data_parallel_size", 1
        ) > 1:
            _dp_parallel = True
            self._dp_parallel = _dp_parallel

    if _dp_parallel is None:
        model_executed = False
        fill_async_mtp_placeholders = getattr(
            self.scheduler,
            "_uses_async_scheduled_mtp_placeholders",
            lambda: False,
        )()
        deferred_scheduler_output: tuple[
            SchedulerOutput, Future
        ] | None = None

        while (
            len(batch_queue) < self.batch_queue_size
            and self.scheduler.has_requests()
            and not self._has_unresolved_edge_cloud_draft_parent()
        ):
            self._drain_pd_channel_inbox()
            scheduler_output = self.scheduler.schedule()
            self._ensure_pd_head_token(scheduler_output)

            # [ascend insert] Publish head-segment batches immediately at
            # schedule time to keep the pipeline full.
            if scheduler_output.batch_type in (
                BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST, BatchType.DRAFT_FIRST
            ):
                self._maybe_publish_pre_out(scheduler_output)

            if scheduler_output.batch_type == BatchType.EMPTY:
                if batch_queue:
                    self._defer_empty_batch(scheduler_output)
                    break
                return self._finish_empty_batch(scheduler_output)

            self._merge_pending_worker_cleanup(scheduler_output)
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )

            scheduled_model_executed = False
            if self.is_ec_consumer:
                scheduled_model_executed = (
                    scheduler_output.total_num_scheduled_tokens > 0
                )
                model_executed |= scheduled_model_executed

            if self.is_pooling_model or not scheduled_model_executed:
                future = cast(Future[ModelRunnerOutput], exec_future)
            elif not self._needs_sample_tokens(scheduler_output):
                future = cast(Future[ModelRunnerOutput], exec_future)
            elif not scheduler_output.pending_structured_output_tokens:
                grammar_output = self.scheduler.get_grammar_bitmask(
                    scheduler_output
                )
                future = self.model_executor.sample_tokens(
                    grammar_output, non_block=True
                )
            else:
                # This execute must remain ordered in the worker MQ, but sampling
                # waits until the prior async output updates grammar state.
                deferred_scheduler_output = (
                    scheduler_output,
                    cast(Future, exec_future),
                )
                break

            batch_queue.appendleft((future, scheduler_output, exec_future))
            queue_types = [
                so.batch_type.value
                for _, so, _ in batch_queue
            ]
            vllm_logger.info(
                "[BATCH_QUEUE] Enqueued %s, queue_len=%d, types=%s",
                scheduler_output.batch_type.value,
                len(batch_queue),
                queue_types,
            )
            if not fill_async_mtp_placeholders:
                # Preserve the upstream one-schedule-per-turn behavior for every
                # other mode. Only async scheduled-MTP needs one EngineCore turn
                # to materialize the complete placeholder chain.
                if (
                    scheduled_model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    return None, True
                break

        if not batch_queue:
            # No completed/in-flight batch is available to collect. This can
            # happen while waiting for a remote prefill tail.
            return None, model_executed

    else:
        model_executed = False
        deferred_scheduler_output = None
        # [ascend insert] Pull cloud-returned tail-segment batches into the
        # scheduler ready queues BEFORE coord/intent. This MUST run every step
        # (unconditional, not gated on _should_schedule): when the edge is
        # waiting for a cloud tail, prefills_last_ready is empty until drained,
        # so _intended_batch_type() returns EMPTY -> winner EMPTY ->
        # _should_schedule False. Gating the drain on _should_schedule would
        # skip it forever -> PL/DL never drained -> prefill/decode never
        # completes (deadlock). Draining first breaks the cycle: PL/DL enters
        # the ready queue, intended reflects it, winner becomes non-EMPTY.
        self._drain_pd_channel_inbox()
        # [ascend insert] Cross-DP batch_type coordination (MoE DP>1): exchange
        # intended batch_type every step and force-schedule the agreed winner so
        # both DPs execute the same batch_type -> edge [0,5] and cloud EP
        # all-toall pair on the same layer. engines_running guarantees both DPs
        # reach here, and the blocking all_reduce keeps iterations 1:1 (see
        # _coordinate_bt). The idle DP (no local requests) still enters the
        # schedule branch via the winner (!= EMPTY) and produces a dummy.
        _coordinated = self._is_coordinated_dp()
        _coord_winner = BatchType.EMPTY
        if _coordinated:
            _intended_batch_type = self.scheduler._intended_batch_type()
            self.step_cnt += 1
            # Combined coord + has_unfinished all_reduce: also exchanges a
            # "has work" flag so engines_running is recomputed every step on
            # both DPs (see _coordinate_bt). Store the result for the busy loop's
            # _has_global_unfinished_reqs to reuse (no separate has_unfinished
            # all_reduce in coord mode).
            #
            # IMPORTANT: use has_requests() (NOT has_unfinished_requests()).
            # has_requests() = has_unfinished_requests() OR has_finished_requests()
            # - it includes finished-but-not-yet-returned requests that still need
            # a step to be cleaned up. Combined with bool(batch_queue) (in-flight
            # batches whose forward may be done but not popped), this covers all
            # reasons a DP must keep stepping. If engines_running only reflected
            # has_unfinished_requests, a DP with a finished-not-returned request
            # (has_work=True via has_requests) would loop into coord while the
            # peer (no work) paused -> coord blocks waiting for the paused peer
            # -> hang. Exchanging has_requests()|batch_queue keeps both DPs
            # running until ALL work (including finished-request cleanup and
            # batch_queue drain) is done on both sides.
            _has_req = self.scheduler.has_requests()
            _has_bq = bool(self.batch_queue)
            _local_has_work = _has_req or _has_bq
            # has_work_waiting: this DP has real decode work (running reqs) but
            # intended EMPTY this step -> it is waiting for its DECODE_LAST to
            # return from cloud, NOT truly idle. Lets _coordinate_bt defer a
            # peer's DECODE_FIRST (Rule 1) so both DPs align on DF instead of
            # spinning one-real-one-dummy. running[] empty + EMPTY == truly
            # idle -> False (peer keeps default DF + dummy to pair the a2a).
            _has_work_waiting = (
                bool(getattr(self.scheduler, "running", None))
                and _intended_batch_type == BatchType.EMPTY
            )
            _coord_winner, _coord_engines_running = self._coordinate_bt(
                _intended_batch_type, _local_has_work, _has_work_waiting
            )
            vllm_logger.info(f"step_cnt={self.step_cnt} _coord_winner={_coord_winner} _intended_batch_type={_intended_batch_type}")
            self._coord_engines_running = _coord_engines_running
            _cnt = getattr(self, "_coord_bt_count", 0)
            if _local_has_work or _cnt % 32 == 0:
                vllm_logger.info(
                    "[DPDBG][COORD-IN] dp_rank=%s has_requests=%s has_unfinished=%s "
                    "batch_queue=%s local_has_work=%s winner=%s engines_running=%s",
                    self.vllm_config.parallel_config.data_parallel_rank,
                    int(_has_req),
                    int(self.scheduler.has_unfinished_requests()),
                    int(_has_bq), int(_local_has_work),
                    _BT_COORD_ID.get(_coord_winner, 0), int(_coord_engines_running),
                )
        # In coord mode, also schedule when has_requests() even if winner=EMPTY:
        # a finished-but-not-yet-returned request (has_requests=True,
        # has_unfinished=False) needs an empty batch to carry its
        # finished_req_ids through update_from_output so it gets cleaned up.
        # Without this, winner=EMPTY skips scheduling -> finished_req_ids never
        # returned -> has_requests stays True forever -> this DP loops in coord
        # forever (and blocks the peer). _schedule_target(EMPTY) returns
        # _make_empty_batch() which carries finished_req_ids.
        _should_schedule = bool(
            (_coordinated and (_coord_winner != BatchType.EMPTY or _has_req))
            or ((not _coordinated) and self.scheduler.has_requests())
        )
        if _should_schedule:
            if _coordinated:
                scheduler_output = self.scheduler._schedule_target(_coord_winner)
            else:
                scheduler_output = self.scheduler.schedule()
            self._hang_last_bt = str(scheduler_output.batch_type)

            # [ascend insert] Assign head-token for edge-cloud head-segment
            # batches so the tail-segment can be matched to the suspended
            # state.
            self._ensure_pd_head_token(scheduler_output)

            # [ascend insert] Publish head-segment batches immediately at
            # schedule time to keep the pipeline full.
            if scheduler_output.batch_type in (
                BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST, BatchType.DRAFT_FIRST
            ):
                self._maybe_publish_pre_out(scheduler_output)
            elif scheduler_output.batch_type in (
                BatchType.PREFILL_LAST, BatchType.DECODE_LAST
            ):
                # [方案③-fix Part 2] tail segment is edge-local (no real cloud
                # zmq). In the NON-coordinated path the peer DP's dummy goes to
                # cloud, so publish a dummy-middle zmq to keep the cloud-side
                # cross-DP all_reduce pairing 1:1.
                # Under cross-DP coordination BOTH DPs run the same tail bt this
                # step, so both clouds are idle (tail is edge-only) and pair
                # naturally - skip the dummy zmq (publishing it would only desync
                # the cloud PassiveScheduler state machine).
                _dp_gt1 = getattr(
                    self.vllm_config.parallel_config, 'data_parallel_size', 1
                ) > 1
                if _dp_gt1 and not _coordinated:
                    self._publish_pd_dummy_zmq()

            if scheduler_output.batch_type == BatchType.EMPTY:
                if batch_queue:
                    self._defer_empty_batch(scheduler_output)
                    scheduler_output = None
                else:
                    return self._finish_empty_batch(scheduler_output)

            if scheduler_output is not None:
                self._merge_pending_worker_cleanup(scheduler_output)

                with self.log_error_detail(scheduler_output):
                    exec_future = self.model_executor.execute_model(
                        scheduler_output, non_block=True
                    )
                if self.is_ec_consumer:
                    model_executed = (
                        scheduler_output.total_num_scheduled_tokens > 0
                    )

                if self.is_pooling_model or not model_executed:
                    # No sampling required (no requests scheduled).
                    future = cast(Future[ModelRunnerOutput], exec_future)
                elif not self._needs_sample_tokens(scheduler_output):
                    # [ascend insert] Edge-cloud head segment (PF/DF): sampling is
                    # done in the tail segment (PL/DL) after the cloud returns
                    # intermediate tensors. Skip sample_tokens for the head
                    # segment.
                    future = cast(Future[ModelRunnerOutput], exec_future)
                else:
                    if not scheduler_output.pending_structured_output_tokens:
                        grammar_output = self.scheduler.get_grammar_bitmask(
                            scheduler_output
                        )
                        future = self.model_executor.sample_tokens(
                            grammar_output, non_block=True
                        )
                    else:
                        deferred_scheduler_output = scheduler_output

                if not deferred_scheduler_output:
                    batch_queue.appendleft((future, scheduler_output, exec_future))
                    # [ascend insert] Log batch_queue contents for debugging.
                    queue_types = [
                        so.batch_type.value
                        for _, so, _ in batch_queue
                    ]
                    vllm_logger.info(
                        "[PP-EVT][BATCH_QUEUE] step_cnt=%d Enqueued %s, queue_len=%d, types=%s",
                        self.step_cnt,
                        scheduler_output.batch_type.value,
                        len(batch_queue),
                        queue_types,
                    )
                    if (
                        model_executed
                        and len(batch_queue) < self.batch_queue_size
                        and not batch_queue[-1][0].done()
                    ):
                        return None, True
        elif not batch_queue:
            return None, False

    # Block until the next result is available.
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    bt = scheduler_output.batch_type
    vllm_logger.info(
        "[PD] EngineCore blocking on future.result(): "
        "batch_type=%s total_tokens=%d",
        bt.value if bt else "N/A",
        scheduler_output.total_num_scheduled_tokens)
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

    # Register the deferred draft before abort/model completion can free
    # requests referenced by this parent batch.
    self._register_edge_cloud_draft_parent(scheduler_output, model_output)
    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    self._advance_edge_cloud_draft(scheduler_output, model_output)
    self._clear_pending_edge_cloud_draft_for_finished_requests()

    if deferred_empty_batch := self._pop_deferred_empty_batch():
        empty_outputs, _ = self._finish_empty_batch(deferred_empty_batch)
        if empty_outputs:
            if engine_core_outputs:
                for client_index, output in empty_outputs.items():
                    existing = engine_core_outputs.get(client_index)
                    if existing is None:
                        engine_core_outputs[client_index] = output
                    elif output.finished_requests:
                        existing_finished = existing.finished_requests or set()
                        existing.finished_requests = existing_finished.union(
                            output.finished_requests
                        )
            else:
                engine_core_outputs = empty_outputs

    if deferred_scheduler_output is not None:
        deferred_output, deferred_exec_future = deferred_scheduler_output
        if (
            self.use_spec_decode
            and not self._uses_scheduled_edge_cloud_draft()
        ):
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_output
                )
        grammar_output = self.scheduler.get_grammar_bitmask(
            deferred_output
        )
        deferred_future = self.model_executor.sample_tokens(
            grammar_output, non_block=True
        )
        batch_queue.appendleft(
            (
                deferred_future,
                deferred_output,
                deferred_exec_future,
            )
        )

    return engine_core_outputs, model_executed


# =======================================================================#
# EngineCore.shutdown — close PD/ZMQ resources before stopping the rest.  #
# =======================================================================#
@functools.wraps(_ORIG_ENGINE_CORE_SHUTDOWN)
def _patched_engine_core_shutdown(self):
    ch = getattr(self, "_pp_pd_channel", None)
    if ch is not None:
        try:
            ch.shutdown()
        except Exception:
            logger.exception(
                "Error while shutting down PD-separation ZMQ channel"
            )
        self._pp_pd_channel = None

    _ORIG_ENGINE_CORE_SHUTDOWN(self)


# =======================================================================#
# EngineCoreProc.run_engine_core — keep child-process patch import.       #
# =======================================================================#
def _patched_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0,
                             **kwargs):
    """Delegate to upstream while keeping this patch module as the process
    target so spawn-based child processes import and install the patches.
    """
    return _ORIG_RUN_ENGINE_CORE(
        *args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs
    )


_patched_run_engine_core.__module__ = __name__
_patched_run_engine_core.__qualname__ = "_patched_run_engine_core"


# =======================================================================#
# EngineCoreProc._process_input_queue — full replacement to add the       #
# edge-cloud idle-block branch.                                            #
# =======================================================================#
# Imports kept inside the function-scope dict to mirror the upstream
# module-level imports (`queue`, `DEBUG`) without polluting our patch
# module's top-level namespace.
import queue as _queue_mod  # noqa: E402
import time as _time  # noqa: E402
from logging import DEBUG as _DEBUG  # noqa: E402


def _patched_process_engine_step(self) -> bool:
    """Avoid adding a 1 ms bubble while async batches are still in flight."""
    outputs, model_executed = self.step_fn()
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)
    self.post_step(model_executed)
    async_mtp_in_flight = bool(self.batch_queue) and getattr(
        self.scheduler,
        "_uses_async_scheduled_mtp_placeholders",
        lambda: False,
    )()
    if (
        not model_executed
        and self.scheduler.has_unfinished_requests()
        and not async_mtp_in_flight
    ):
        _time.sleep(0.001)
    return model_executed


def _patched_process_input_queue(self):
    """Exits when an engine step needs to be performed."""
    waited = False
    _piq_rank = getattr(self, "dp_rank", "?")
    logger.info(
        "[HANG] _process_input_queue ENTER: dp_rank=%s has_work=%s has_unfinished=%s "
        "batch_queue=%s engines_running=%s input_empty=%s",
        _piq_rank, self.has_work(), self.scheduler.has_unfinished_requests(),
        bool(self.batch_queue), self.engines_running, self.input_queue.empty(),
    )
    while not self.has_work() and self.is_running():
        # Notify callbacks waiting for engine to become idle.
        self._notify_idle_state_callbacks()
        if self.input_queue.empty():
            with self.aborts_queue.mutex:
                self.aborts_queue.queue.clear()
            if logger.isEnabledFor(_DEBUG):
                logger.debug("EngineCore waiting for work.")
                waited = True
        block = self.process_input_queue_block

        # [ascend insert] In edge-cloud mode the edge can be completely
        # idle for long periods while waiting for the next client
        # request. If no local work exists, force a blocking wait even
        # if an earlier mode (e.g. elastic scaling) left the input queue
        # in non-blocking polling mode; otherwise the outer busy loop
        # spins forever.
        if (
            not block
            and not self.scheduler.has_unfinished_requests()
            and not self.engines_running
            and not bool(self.batch_queue)
            and getattr(self, "eep_scaling_state", None) is None
        ):
            block = True

        logger.info(
            "[HANG] _process_input_queue WAIT: dp_rank=%s has_unfinished=%s "
            "batch_queue=%s engines_running=%s block=%s input_empty=%s",
            _piq_rank, self.scheduler.has_unfinished_requests(),
            bool(self.batch_queue), self.engines_running, block, self.input_queue.empty(),
        )
        try:
            if block and self.input_queue.empty():
                logger.info("input_queue is empty, EngineCore waiting for work.")
            req = self.input_queue.get(block=block)
            self._handle_client_request(*req)
        except _queue_mod.Empty:
            break
        if not block:
            break

    if waited:
        logger.debug("EngineCore loop active.")

    # Handle any more client requests.
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)


# =======================================================================#
# EngineCore.execute_dummy_batch - 方案③: route dummy per-DP via zmq.       #
# =======================================================================#
def _patched_execute_dummy_batch(self):
    """PD-separation edge: mirror execute_model's per-DP zmq path so the
    idle DP's dummy does NOT reach the cloud via the cross-node
    rpc_broadcast_mq broadcast (which would deliver it to the DP running
    real work and break cross-DP all_reduce pairing -> deadlock).

    Cloud workers skip the cross-node ``execute_dummy_batch`` (see
    multiproc_executor.worker_busy_loop), so ``executor.execute_dummy_batch``
    below only runs the dummy on *this* DP's edge workers. We additionally
    publish a dummy SchedulerOutput via zmq so the paired cloud DP runs a
    dummy-middle (see worker._execute_model_cloud `is_pd_dummy` branch) and
    keeps the cloud-side cross-DP all_reduce paired.

    Publish the zmq dummy for EVERY edge dummy - including wave-fill dummies
    that fire while this DP has unfinished requests (``has_unfinished=True``)
    but no real work this wave step. Without this, the real DP's edge skips
    the zmq publish for its wave-fill dummies while the idle DP's edge
    publishes for all of its dummies -> the idle cloud receives N more
    dummies than the real cloud processes -> N dummies backlog on the idle
    cloud -> the next request on the idle DP hangs (real PRE_OUT stuck
    behind the backlog, PP isend init timeout). Publishing here makes cloud
    dummy publication symmetric so both clouds process the same count.

    When PD-separation is off (no ``_pp_pd_channel``) this is identical to
    upstream: just ``executor.execute_dummy_batch()``.
    """
    ch = getattr(self, "_pp_pd_channel", None)
    # Publish dummy zmq whenever dp>1: covers both idle dummies AND wave-fill
    # dummies during a request (has_unfinished=True). For dp=1 there is no
    # peer DP and publishing corrupts the cloud's PassiveScheduler, so skip.
    _dp_gt1 = getattr(self.vllm_config.parallel_config, 'data_parallel_size', 1) > 1
    if ch is not None and _dp_gt1:
        from vllm.v1.core.sched.output import (
            BatchType as _BatchType,
            HiddenChannelType as _HiddenChannelType,
            SchedulerOutput as _SchedulerOutput,
        )
        dummy_so = _SchedulerOutput.make_empty()
        dummy_so.batch_type = _BatchType.DECODE_FIRST
        dummy_so.hidden_channel = _HiddenChannelType.DECODE
        dummy_so.head_token = uuid4().hex
        self._ensure_pd_original_seq(dummy_so)
        # Dynamic marker consumed by cloud _execute_model_cloud / PassiveEngineCore.step.
        setattr(dummy_so, "is_pd_dummy", True)
        ch.publish(dummy_so)
    self.model_executor.execute_dummy_batch()


def _publish_pd_dummy_zmq(self):
    """Publish a dummy-middle zmq to the paired cloud DP (no edge dummy run).

    Used by tail segments (PL/DL) which are edge-local: they don't send a
    real cloud zmq, but the peer DP's dummy goes to cloud. To keep the
    cloud-side cross-DP all_reduce pairing 1:1, the tail step publishes a
    dummy-middle zmq so the paired cloud DP runs a dummy-middle (see
    worker._execute_model_cloud `is_pd_dummy` branch).
    """
    ch = getattr(self, "_pp_pd_channel", None)
    if ch is None:
        return
    from vllm.v1.core.sched.output import (
        BatchType as _BatchType,
        HiddenChannelType as _HiddenChannelType,
        SchedulerOutput as _SchedulerOutput,
    )
    dummy_so = _SchedulerOutput.make_empty()
    dummy_so.batch_type = _BatchType.DECODE_FIRST
    dummy_so.hidden_channel = _HiddenChannelType.DECODE
    dummy_so.head_token = uuid4().hex
    self._ensure_pd_original_seq(dummy_so)
    setattr(dummy_so, "is_pd_dummy", True)
    ch.publish(dummy_so)


# =======================================================================#
# Install                                                                  #
# =======================================================================#
def install() -> None:
    if getattr(EngineCore, _INSTALLED_FLAG, False):
        return

    EngineCore.__init__ = _patched_engine_core_init
    EngineCore._drain_pd_channel_inbox = _drain_pd_channel_inbox
    EngineCore._maybe_publish_pre_out = _maybe_publish_pre_out
    EngineCore._release_deferred_draft_pre_out = (
        _release_deferred_draft_pre_out
    )
    EngineCore._close_draft_pre_out = _close_draft_pre_out
    EngineCore._ensure_pd_head_token = _ensure_pd_head_token
    EngineCore._ensure_pd_original_seq = _ensure_pd_original_seq
    EngineCore._needs_sample_tokens = _needs_sample_tokens
    EngineCore._stash_empty_worker_cleanup = _stash_empty_worker_cleanup
    EngineCore._merge_pending_worker_cleanup = _merge_pending_worker_cleanup
    EngineCore._finish_empty_batch = _finish_empty_batch
    EngineCore._defer_empty_batch = _defer_empty_batch
    EngineCore._pop_deferred_empty_batch = _pop_deferred_empty_batch
    EngineCore._advance_edge_cloud_draft = _advance_edge_cloud_draft
    EngineCore._clear_pending_edge_cloud_draft_for_finished_requests = (
        _clear_pending_edge_cloud_draft_for_finished_requests
    )
    EngineCore._register_edge_cloud_draft_parent = (
        _register_edge_cloud_draft_parent
    )
    EngineCore._uses_scheduled_edge_cloud_draft = (
        _uses_scheduled_edge_cloud_draft
    )
    EngineCore._has_unresolved_edge_cloud_draft_parent = (
        _has_unresolved_edge_cloud_draft_parent
    )
    EngineCore._is_coordinated_dp = _is_coordinated_dp
    EngineCore._coordinate_bt = _coordinate_bt
    EngineCore.step = _patched_step
    EngineCore.step_with_batch_queue = _patched_step_with_batch_queue
    EngineCore.execute_dummy_batch = _patched_execute_dummy_batch
    EngineCore._publish_pd_dummy_zmq = _publish_pd_dummy_zmq
    EngineCore.shutdown = _patched_engine_core_shutdown

    EngineCoreProc.run_engine_core = staticmethod(_patched_run_engine_core)
    EngineCoreProc._process_input_queue = _patched_process_input_queue
    EngineCoreProc._process_engine_step = _patched_process_engine_step

    setattr(EngineCore, _INSTALLED_FLAG, True)
    logger.info(
        "vllm-ascend EngineCore PD/edge-cloud patch installed."
    )


install()
