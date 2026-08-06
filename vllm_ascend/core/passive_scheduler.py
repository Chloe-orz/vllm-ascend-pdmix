# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Passive scheduler for non-leader PP ranks.

A `PassiveScheduler` does not make scheduling decisions. It receives
SchedulerOutputs that have already been decided by the leader rank (rank 0)
over a ZMQ subscriber, classifies them by `batch_type`, and emits ready-to-
dispatch payloads — optionally splitting prefill / PD-mix batches into N
layer slices based on a YAML config.

The class is intentionally minimal: it shares no implementation with
`vllm.v1.core.sched.scheduler.Scheduler` and depends only on the public
`SchedulerOutput` / `BatchType` types.
"""
import enum
import math
import os

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional

from vllm import envs
from vllm.logger import logger
from vllm.v1.core.sched.output import BatchType, SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.core import PPSchedulerZmqSubscriber


class DispatchPolicy(enum.Enum):
    """Order in which phase queues are drained inside :meth:`PassiveScheduler.schedule`.

    The three phase queues — PURE_PREFILL, PD_MIX, PURE_DECODE — are polled
    in the order encoded by the policy. One SchedulerOutput is picked per
    non-empty queue per call.
    """
    EXPECT_ALTERNATION = "expect_alternation"  # Phase7 EEP/EED state machine.
    PREFILL_FIRST = "prefill_first"   # P  → PD-mix → D
    DECODE_FIRST = "decode_first"     # D  → PD-mix → P
    PDMIX_FIRST = "pdmix_first"       # PD-mix → P → D


class CloudSchedulingState(enum.Enum):
    EXPECT_EXECUTE_PREFILL = "expect_execute_prefill"
    EXPECT_EXECUTE_DECODE_OR_DRAFT = "expect_execute_decode_or_draft"


@dataclass
class LayerSliceInfo:
    """Metadata for a single layer slice in layerwise-disaggregated execution.

    When layer slicing is enabled, the PassiveScheduler splits the local
    layer range of a single SchedulerOutput into N slices. Each slice carries
    this info so the worker / model_runner can run only the assigned layer
    range and decide whether to perform PP communication.
    """
    slice_index: int       # 0, 1, 2, ...
    total_slices: int      # N
    start_layer: int       # local start layer (0-based within local layers)
    end_layer: int         # local end layer
    is_first_slice: bool   # slice_index == 0
    is_last_slice: bool    # slice_index == total_slices - 1


@dataclass
class ScheduledBatch:
    """Output of `PassiveScheduler.schedule()`: one SchedulerOutput plus the
    layer slices to dispatch in this engine tick.

    - For PURE_DECODE / DECODE_FIRST batches, or when slicing is disabled:
      ``slices == [None]`` (single full-layer execution).
    - For sliced PURE_PREFILL / PREFILL_FIRST / PD_MIX batches, ``schedule()``
      returns a single ``LayerSliceInfo`` per call so decode batches can be
      interleaved between prefill middle-layer slices.

    An empty instance (``slices == []``) signals that no SchedulerOutput was
    available to dispatch this round; the caller should typically idle.
    """
    scheduler_output: SchedulerOutput
    slices: list["LayerSliceInfo | None"]

    @classmethod
    def empty(cls) -> "ScheduledBatch":
        return cls(scheduler_output=None, slices=[])  # type: ignore[arg-type]

    def is_empty(self) -> bool:
        return not self.slices


class SliceTask(NamedTuple):
    scheduler_output: SchedulerOutput
    slice_info: LayerSliceInfo | None


class PassiveScheduler:
    """Receive → classify → schedule, for non-leader PP ranks.

    Lifecycle (each tick of the engine-core main loop):

        passive_scheduler.poll_and_classify()
        batch = passive_scheduler.schedule()
        if not batch.is_empty():
            for slice_info in batch.slices:
                executor.rpc_broadcast_mq.enqueue(...)

    `schedule()` returns a `ScheduledBatch` with 1 SchedulerOutput plus
    the slice plan; a single PURE_PREFILL / PD_MIX batch may carry N
    layer slices, while PURE_DECODE / DECODE_FIRST batches always carry
    `[None]` (single full-layer execution).
    """

    _ARRIVAL_SEQ_ATTR = "_passive_scheduler_arrival_seq"

    def __init__(
        self,
        vllm_config: "VllmConfig",
        pp_subscriber: "PPSchedulerZmqSubscriber",
        dispatch_policy: DispatchPolicy = DispatchPolicy.EXPECT_ALTERNATION,
        run_subscriber_thread: bool = True,
        dp_coord_group=None,  # stateless ProcessGroup for cross-DP coordination
    ) -> None:
        self.pp_subscriber = pp_subscriber
        self.vllm_config = vllm_config
        self.dispatch_policy = dispatch_policy
        self.cloud_scheduling_state = CloudSchedulingState.EXPECT_EXECUTE_PREFILL

        # Optional cross-DP coordination group. Set by the engine core
        # when dp>1 + MoE + PD-separation.  schedule() uses it to
        # coordinate batch_type decisions across cloud DPs.
        self.dp_coord_group = dp_coord_group

        self.ready_prefills: deque[SchedulerOutput] = deque()
        self.ready_pdmixes: deque[SchedulerOutput] = deque()
        self.ready_drafts: deque[SchedulerOutput] = deque()
        self.ready_decodes: deque[SchedulerOutput] = deque()

        # Active sliced prefill / PD-mix continuation.  Only one sliced
        # prefill-like batch is allowed to be active at a time because the
        # Ascend model runner keeps layerwise continuation state in single
        # ``_layerwise_*`` fields.  Decode batches may be interleaved between
        # these continuation slices; another prefill-like slice-0 may not.
        self._active_sliced_prefill: SchedulerOutput | None = None
        self._active_prefill_slices: deque[SliceTask] = deque()

        # Coordinated mode: pre-synced total slice count for the
        # ready_prefills head, aligned across DPs via all_reduce(MAX)
        # before DP0 executes (see _schedule_expect_alternation).  Lets a
        # dummy on one DP slice the same count as the peer's real prefill
        # so d_slices stays in sync.  None outside coordinated mode.
        self._coordinated_total_slices: int | None = None

        # Cloud-side P/D interleave guard. After dispatching one prefill-middle
        # slice, EXPECT_EXECUTE_DECODE_OR_DRAFT waits up to 10ms for a decode-middle
        # batch before falling back to another prefill-middle slice.
        self._prefill_middle_throttle_started_at: float | None = None
        self._prefill_middle_throttle_seconds = 0.010

        # Bridge queue between the (optional) subscriber thread and the
        # main loop. When the thread is enabled, it drains
        # `pp_subscriber.consume_new_outputs()` and pushes each SchedulerOutput
        # into `_inbox`; `poll_and_classify` drains `_inbox` instead of
        # touching the subscriber directly.
        self._inbox: queue.Queue[tuple[int, SchedulerOutput]] = queue.Queue()
        self._subscriber_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # [DIAG] Track DECODE_FIRST arrival intervals on the cloud side.
        self._last_decode_first_arrival_ts: float | None = None

        # [DIAG] Engine-tick counter; incremented once per schedule() call so
        # slice / DP decisions can be correlated across logs by step number.
        self._step: int = 0

        # Precompute local layer count.  The actual slice count is resolved
        # per-batch from a YAML config (token threshold -> slice count).
        self._num_local_layers = 0
        self._layer_slice_config: dict[int, int] | None = None
        self._layer_slice_config_mtime: float = 0.0
        self._layer_slice_config_path: str | None = None
        # Use hf_text_config (not the root hf_config) so multimodal models
        # whose layer count lives in a nested text sub-config (e.g. KimiK2.5
        # -> DeepseekV3Config) resolve correctly. For plain-text models
        # hf_text_config == hf_config, so this is a no-op there.
        num_hidden_layers = (
            vllm_config.model_config.hf_text_config.num_hidden_layers
        )
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if vllm_config.parallel_config.enable_edge_cloud:
            head_k = tail_k = 1
            additional_config = getattr(
                vllm_config, "additional_config", None
            )
            if isinstance(additional_config, dict):
                ec_cfg = additional_config.get("edge_cloud_config", {})
                htl = ec_cfg.get("edge_head_tail_layers", 1)
                if isinstance(htl, int):
                    head_k = tail_k = htl
                elif isinstance(htl, (list, tuple)) and len(htl) >= 2:
                    head_k = int(htl[0])
                    tail_k = int(htl[1])
            self._num_local_layers = max(
                0, num_hidden_layers - head_k - tail_k
            )
        else:
            from vllm.distributed.utils import get_pp_indices
            start_layer_pp, end_layer = get_pp_indices(
                num_hidden_layers, pp_size - 1, pp_size
            )
            self._num_local_layers = end_layer - start_layer_pp

        if self._num_local_layers > 0:
            cfg = self._load_layer_slice_config()
            if cfg is not None:
                self._layer_slice_config = cfg
                logger.info(
                    f"[PassiveScheduler] Layer-slice config loaded: "
                    f"{self._layer_slice_config}",
                )
            else:
                logger.info(
                    "[PassiveScheduler] Layer-slice YAML not found; "
                    "layer slicing is disabled.",
                )

        if run_subscriber_thread:
            self.start_subscriber_thread()

    # ------------------------------------------------------------------ #
    # Subscriber thread lifecycle                                        #
    # ------------------------------------------------------------------ #
    def start_subscriber_thread(self) -> None:
        """Spawn a daemon thread that pulls from the ZMQ subscriber and
        pushes SchedulerOutputs into `_inbox`. Idempotent.
        """
        if self._subscriber_thread is not None:
            return
        self._shutdown_event.clear()
        self._subscriber_thread = threading.Thread(
            target=self._subscriber_loop,
            name="PassiveScheduler-Subscriber",
            daemon=True,
        )
        self._subscriber_thread.start()
        logger.debug("PassiveScheduler subscriber thread started.")

    def _subscriber_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                new_outputs = self.pp_subscriber.consume_new_outputs()
            except Exception:
                if self._shutdown_event.is_set():
                    return
                logger.exception(
                    "PassiveScheduler subscriber thread failed to consume."
                )
                return
            if not new_outputs:
                # Avoid a tight spin when the subscriber returns nothing.
                self._shutdown_event.wait(0.001)
                continue
            for seq, scheduler_output in new_outputs:
                self._inbox.put((seq, scheduler_output))

    def shutdown(self) -> None:
        """Signal the subscriber thread to stop and join it."""
        self._shutdown_event.set()
        thread = self._subscriber_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._subscriber_thread = None

    # ------------------------------------------------------------------ #
    # Inbox draining + classification                                    #
    # ------------------------------------------------------------------ #
    def poll_and_classify(self) -> None:
        """Drain SchedulerOutputs from the inbox (fed by the subscriber
        thread, or directly by `_drain_subscriber_inline` when the thread
        is disabled) and route each into its phase-specific ready queue.
        """
        if self._subscriber_thread is None:
            # Inline mode: pull from the subscriber directly into _inbox.
            self._drain_subscriber_inline()

        while True:
            try:
                seq, scheduler_output = self._inbox.get_nowait()
            except queue.Empty:
                break
            self._remember_arrival_seq(scheduler_output, seq)
            bt = scheduler_output.batch_type
            # [DPDBG] classify dummy vs real prefill/decode. The dummy zmq
            # (from _patched_execute_dummy_batch / _publish_pd_dummy_zmq) is
            # published with batch_type=DECODE_FIRST but
            # total_num_scheduled_tokens==0 (the is_pd_dummy attr is lost in
            # zmq serialization, so the cloud detects dummy by tokens==0).
            # Distinguish so the dummy-flood vs real-request arrival counts per
            # cloud DP are visible: the dp=2 PP-init-timeout symptom is one
            # cloud DP seeing far more dummies before its real PREFILL_FIRST
            # than the other (e.g. seq=282 vs 68 for the real PREFILL_FIRST).
            _total = scheduler_output.total_num_scheduled_tokens
            if _total == 0:
                _kind = "DUMMY"
            elif bt in (BatchType.PURE_PREFILL, BatchType.PREFILL_FIRST):
                _kind = "REAL_PREFILL"
            elif bt in (BatchType.PURE_DECODE, BatchType.DECODE_FIRST):
                _kind = "REAL_DECODE"
            elif bt in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST):
                _kind = "TAIL(edge-only)"
            else:
                _kind = f"OTHER({bt.value if bt is not None else None})"
            _kc = getattr(self, "_dpdbg_kind_count", None)
            if _kc is None:
                _kc = {}
                self._dpdbg_kind_count = _kc
            _kc[_kind] = _kc.get(_kind, 0) + 1
            _dp_rank = getattr(
                getattr(self.vllm_config, "parallel_config", None),
                "data_parallel_rank", "?",
            )
            logger.info(
                "[DPDBG] PassiveScheduler recv: dp_rank=%s seq=%s kind=%s "
                "batch_type=%s tokens=%s kind_counts=%s",
                _dp_rank, seq, _kind,
                bt.value if bt is not None else None, _total, _kc,
            )
            if bt == BatchType.EMPTY:
                continue
            elif bt in (BatchType.PURE_PREFILL, BatchType.PREFILL_FIRST):
                # PREFILL_FIRST = edge-cloud "P first" head segment; from the
                # cloud's perspective it is exactly the same workload as a
                # legacy PURE_PREFILL batch (run middle layers, send hidden
                # state back), so route into the same ready queue.
                self.ready_prefills.append(scheduler_output)
            elif bt in (BatchType.PURE_DECODE, BatchType.DECODE_FIRST):
                # Same reasoning as above for decode head segments.
                now = time.monotonic()
                if self._last_decode_first_arrival_ts is not None:
                    interval_ms = (now - self._last_decode_first_arrival_ts) * 1000
                    logger.debug(
                        "DECODE_FIRST arrival interval: %.2f ms",
                        interval_ms,
                    )
                self._last_decode_first_arrival_ts = now
                self.ready_decodes.append(scheduler_output)
            elif bt == BatchType.DRAFT_FIRST:
                self.ready_drafts.append(scheduler_output)
            elif bt in (
                BatchType.PREFILL_LAST,
                BatchType.DECODE_LAST,
                BatchType.DRAFT_LAST,
            ):
                # Tail-segment batches are edge-only and must never be
                # dispatched on the cloud. If one shows up here it is a
                # routing bug at the publisher side — drop with a loud log.
                logger.error(
                    "PassiveScheduler received tail-segment batch_type=%s; "
                    "tail segments are edge-only and will be dropped.",
                    bt.value,
                )
                continue
            else:  # PD_MIX (or anything unrecognized — treat as mix)
                self.ready_pdmixes.append(scheduler_output)
            logger.debug(
                "PassiveScheduler classified seq=%s batch_type=%s "
                "(prefills=%d, pdmixes=%d, drafts=%d, decodes=%d)",
                self._arrival_seq(scheduler_output),
                bt.value if bt is not None else "<none>",
                len(self.ready_prefills),
                len(self.ready_pdmixes),
                len(self.ready_drafts),
                len(self.ready_decodes),
            )

    def _remember_arrival_seq(
        self, scheduler_output: SchedulerOutput, seq: int
    ) -> None:
        try:
            setattr(scheduler_output, self._ARRIVAL_SEQ_ATTR, seq)
        except Exception:
            logger.debug(
                "Unable to attach arrival seq=%d to SchedulerOutput.",
                seq,
                exc_info=True,
            )

    def _arrival_seq(self, scheduler_output: SchedulerOutput) -> int | None:
        seq = getattr(scheduler_output, self._ARRIVAL_SEQ_ATTR, None)
        return seq if isinstance(seq, int) else None

    def _drain_subscriber_inline(self) -> None:
        """Used only when the subscriber thread is disabled (e.g. tests)."""
        new_outputs = self.pp_subscriber.consume_new_outputs()
        for seq, scheduler_output in new_outputs:
            self._inbox.put((seq, scheduler_output))

    # ------------------------------------------------------------------ #
    # Layer-slice config loading                                         #
    # ------------------------------------------------------------------ #
    def _load_layer_slice_config(self) -> dict[int, int] | None:
        """Load token-threshold -> slice-count mapping from YAML.

        The YAML is expected to contain entries like::

            16: 24
            8: 10
            4: 4
            1: 5
            0: 5

        where the key is the token count in *thousands* and the value is
        the total number of slices.

        The file path resolution order is:
        1. ``VLLM_LAYER_SLICE_CONFIG`` env var if set.
        2. ``layer_slice_config.yaml`` in the same directory as this module.

        On success the path and mtime are cached on ``self`` for hot-reload
        tracking.  Returns ``None`` when the file does not exist or cannot
        be parsed.
        """
        yaml_path = os.environ.get("VLLM_LAYER_SLICE_CONFIG")
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(__file__), "layer_slice_config.yaml"
            )
        if not os.path.exists(yaml_path):
            self._layer_slice_config_path = None
            self._layer_slice_config_mtime = 0.0
            return None
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                logger.warning(
                    "Layer-slice config %s is not a dict; ignoring.", yaml_path
                )
                return None
            # Extract optional prefill_middle_throttle_ms (milliseconds) before filtering.
            _throttle_key = "prefill_middle_throttle_ms"
            if _throttle_key in raw:
                try:
                    self._prefill_middle_throttle_seconds = float(raw[_throttle_key]) / 1000.0
                    logger.info(
                        "[PassiveScheduler] %s set to %.1f ms (%.3f s) from %s",
                        _throttle_key, float(raw[_throttle_key]),
                        self._prefill_middle_throttle_seconds, yaml_path,
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid %s value %r in %s; keeping %.3f s",
                        _throttle_key, raw[_throttle_key], yaml_path,
                        self._prefill_middle_throttle_seconds,
                    )

            # Normalize to int keys / values and sort descending by token threshold.
            config = {
                int(k): int(v) for k, v in raw.items() if isinstance(k, (int, str)) and str(k).lstrip('-').isdigit()
            }
            self._layer_slice_config_path = yaml_path
            self._layer_slice_config_mtime = os.path.getmtime(yaml_path)
            return dict(sorted(config.items(), key=lambda item: item[0], reverse=True))
        except Exception:
            logger.exception("Failed to load layer-slice config from %s", yaml_path)
            return None

    def _maybe_hot_reload_layer_slice_config(self) -> None:
        """Check whether the YAML config file has changed on disk and reload."""
        path = self._layer_slice_config_path
        if path is None:
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime != self._layer_slice_config_mtime:
            new_cfg = self._load_layer_slice_config()
            if new_cfg is not None:
                self._layer_slice_config = new_cfg
                logger.info(
                    f"[PassiveScheduler] Layer-slice config hot-reloaded: "
                    f"{self._layer_slice_config}",
                )

    def _resolve_slice_count(self, total_tokens: int) -> int:
        """Map a prefill batch size (in tokens) to the desired slice count.

        Uses the loaded YAML config (token-threshold in **thousands** ->
        slice-count).  The thresholds are checked from largest to smallest;
        the first threshold that ``total_tokens`` meets or exceeds wins.

        If no YAML config is present, layer slicing is disabled.
        """
        self._maybe_hot_reload_layer_slice_config()
        if self._layer_slice_config is not None:
            for token_k, slice_num in self._layer_slice_config.items():
                if total_tokens >= token_k * 1000:
                    return slice_num
        return 0

    # ------------------------------------------------------------------ #
    # Slicing                                                            #
    # ------------------------------------------------------------------ #
    def _make_slice_info(
        self,
        slice_idx: int,
        total_slices: int,
        boundaries: list[tuple[int, int]],
    ) -> LayerSliceInfo:
        slice_start, slice_end = boundaries[slice_idx]
        return LayerSliceInfo(
            slice_index=slice_idx,
            total_slices=total_slices,
            start_layer=slice_start,
            end_layer=slice_end,
            is_first_slice=(slice_idx == 0),
            is_last_slice=(slice_idx == total_slices - 1),
        )

    def _do_slice(
        self, so: SchedulerOutput, total_slices: Optional[int] = None,
    ) -> list["LayerSliceInfo | None"]:
        """Compute layer slices for a prefill-like batch."""
        if total_slices is None:
            total_slices = self._resolve_slice_count(
                so.total_num_scheduled_tokens
            )
        # [DIAG] Log the resolved slice count with DP + step context so the
        # per-tick slicing decision can be correlated across cloud DPs.
        _dp_rank = getattr(
            self.vllm_config.parallel_config, "data_parallel_rank", "?"
        )
        _dp_size = getattr(
            self.vllm_config.parallel_config, "data_parallel_size", 1
        )
        logger.info(
            "[SLICE-DIAG] step=%s dp_rank=%s/%s total_slices=%s",
            self._step, _dp_rank, _dp_size, total_slices,
        )
        # Slicing disabled or trivially 1 slice.
        if total_slices <= 1:
            return [None]

        boundaries = self._compute_slice_boundaries(
            self._num_local_layers, total_slices
        )
        return [
            self._make_slice_info(i, total_slices, boundaries)
            for i in range(total_slices)
        ]

    def _slice_for(
        self, so: SchedulerOutput, total_slices: Optional[int] = None,
    ) -> list["LayerSliceInfo | None"]:
        # Decode-like and empty batches are never sliced. DECODE_FIRST is the
        # edge-cloud head segment of a decode step — same per-token shape as
        # PURE_DECODE, so it follows the same no-slice rule.
        if so.batch_type in (
            BatchType.PURE_DECODE,
            BatchType.DECODE_FIRST,
            BatchType.DRAFT_FIRST,
        ):
            if getattr(self, "_step", None):
                logger.debug(
                    "[COORD-DIAG] _slice_for step=%d bt=%s → no-slice (decode/draft type)",
                    self._step, so.batch_type.value if so.batch_type else "?",
                )
            return [None]

        # Coordinated mode pre-synced a slice count across DPs (see
        # _schedule_expect_alternation).  When slicing is warranted below
        # (ready_decodes / cloud_suggest), use this synced count instead of
        # the local so.tokens, so a dummy on DP0 slices the same N as DP1's
        # real prefill (else dummy resolve(0)=1 → d_slices=0 → DP1 real
        # unsliced).  No decode / no suggest still falls through to no-slice
        # (cold-start: nothing to interleave).  None outside coordinated mode.
        if total_slices is None:
            total_slices = getattr(self, "_coordinated_total_slices", None)

        # [方案B] Cloud 侧决策：
        # 1. 已有 decode 到达 Cloud → 强制切层（确定性收益）
        if self.ready_decodes:
            if getattr(self, "_step", None):
                logger.info(
                    "[COORD-DIAG] _slice_for step=%d bt=%s → do_slice (ready_decodes=%d)",
                    self._step, so.batch_type.value if so.batch_type else "?",
                    len(self.ready_decodes),
                )
            return self._do_slice(so, total_slices)

        # 2. Edge 建议切层（decode 正在路上）→ 切层
        #    decision.cloud_suggest_slicing 优先，为 None 时回退到 so.cloud_suggest_slicing
        _cloud_suggest = (
             getattr(so, "cloud_suggest_slicing", False)
        )
        if _cloud_suggest:
            if getattr(self, "_step", None):
                logger.info(
                    "[COORD-DIAG] _slice_for step=%d bt=%s → do_slice (cloud_suggest=%s)",
                    self._step, so.batch_type.value if so.batch_type else "?",
                    _cloud_suggest,
                )
            return self._do_slice(so, total_slices)

        # 3. 协调模式下 pre-sync 了 >1 的切层意图（peer DP 有 decode 需求）
        #    → 即使本地无 decode / cloud_suggest 也切层，否则 peer 的 real
        #    prefill 会被本 DP 的 dummy（d_slices=0）强制不切。冷启动时意图=0，
        #    此分支不命中，仍走下面的 no-slice。
        if total_slices is not None and total_slices > 1:
            if getattr(self, "_step", None):
                logger.info(
                    "[COORD-DIAG] _slice_for step=%d bt=%s → do_slice "
                    "(coordinated_total_slices=%d)",
                    self._step, so.batch_type.value if so.batch_type else "?",
                    total_slices,
                )
            return self._do_slice(so, total_slices)

        # 4. Edge 建议不切层 + Cloud 无 decode → 明确不切层（冷启动优化）
        # 短 prefill（<8k）执行太快，decode 来不及穿插，同样不切层
        if getattr(self, "_step", None):
            logger.info(
                "[COORD-DIAG] _slice_for step=%d bt=%s → no-slice (no decode, no suggest)",
                self._step, so.batch_type.value if so.batch_type else "?",
            )
        return [None]

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    _POLICY_ORDER: dict[DispatchPolicy, tuple[str, str, str]] = {
        DispatchPolicy.EXPECT_ALTERNATION: (
            "ready_prefills", "ready_decodes", "ready_pdmixes",
        ),
        DispatchPolicy.PREFILL_FIRST: (
            "ready_prefills", "ready_pdmixes", "ready_decodes",
        ),
        DispatchPolicy.DECODE_FIRST: (
            "ready_decodes", "ready_pdmixes", "ready_prefills",
        ),
        DispatchPolicy.PDMIX_FIRST: (
            "ready_pdmixes", "ready_prefills", "ready_decodes",
        ),
    }

    def _start_prefill_middle_throttle(self) -> None:
        self._prefill_middle_throttle_started_at = time.monotonic()
        logger.info(
            f"[PD-PASSIVE] Prefill throttle started: waiting up to "
            f"{self._prefill_middle_throttle_seconds * 1000:.0f}ms for decode",
        )

    def _clear_prefill_middle_throttle(self) -> None:
        started_at = self._prefill_middle_throttle_started_at
        if started_at is not None:
            elapsed_ms = (time.monotonic() - started_at) * 1000
            logger.info(
                f"[PD-PASSIVE] Prefill throttle cleared after "
                f"{elapsed_ms:.1f}ms",
            )
        self._prefill_middle_throttle_started_at = None

    def _can_fallback_to_prefill_in_decode_state(self) -> bool:
        # Optimization: when the next prefill to schedule has
        # cloud_suggest_slicing=False, the edge signaled "no running decode
        # in flight" -> decode is not about to arrive, so waiting (throttling)
        # for it is pure idle.  Skip the throttle and fall back to prefill
        # immediately.  This also means the P-middle is unsliced (see
        # _slice_for), so there is no slice interleaving to protect either.
        if self.ready_prefills and not getattr(
            self.ready_prefills[0], "cloud_suggest_slicing", False
        ):
            self._clear_prefill_middle_throttle()
            return True
        started_at = self._prefill_middle_throttle_started_at
        if started_at is None:
            return True
        elapsed_ms = (time.monotonic() - started_at) * 1000
        limit_ms = self._prefill_middle_throttle_seconds * 1000
        if elapsed_ms >= limit_ms:
            logger.error(
                f"[PD-PASSIVE] Throttle timeout: waited {elapsed_ms:.1f}ms, "
                f"fallback to prefill",
            )
            self._clear_prefill_middle_throttle()
            return True
        logger.debug(
            f"[PD-PASSIVE] Throttle active: {elapsed_ms:.1f}ms / {limit_ms:.0f}ms, "
            f"still waiting for decode",
        )
        return False

    def schedule(self) -> ScheduledBatch:
        """Pick the next SchedulerOutput to dispatch.

        ``EXPECT_ALTERNATION`` implements the Phase7 cloud-side EEP/EED state
        machine.  Sliced prefill-like batches are dispatched one slice per call
        so decode/draft batches can be interleaved between the remaining
        slices.  Draft priority is enforced inside the state machine, not via
        an early out-of-band check.
        """
        # [DIAG] One step per engine tick so slice/DP decisions can be
        # correlated across logs.  Incremented here (the single per-tick
        # entry point) regardless of which dispatch path is taken.
        self._step += 1
        if self.dispatch_policy == DispatchPolicy.EXPECT_ALTERNATION:
            return self._schedule_expect_alternation()

        for queue_name in self._POLICY_ORDER[self.dispatch_policy]:
            batch = self._schedule_from_queue(queue_name)
            if not batch.is_empty():
                return batch

        return ScheduledBatch.empty()

    @staticmethod
    def _compute_slice_boundaries(
        num_local_layers: int, layer_slice_num: int
    ) -> list[tuple[int, int]]:
        """Compute layer slice boundaries for a fixed slice count.

        Distributes ``num_local_layers`` into ``layer_slice_num`` slices
        as evenly as possible.  Larger slices come first; the size
        difference between any two slices is at most 1.

        Returns a list of ``(start_layer, end_layer)`` tuples where
        ``end_layer`` is exclusive.
        """
        if num_local_layers <= 0 or layer_slice_num <= 0:
            return []
        boundaries: list[tuple[int, int]] = []
        base = num_local_layers // layer_slice_num
        rem = num_local_layers % layer_slice_num
        start = 0
        for i in range(layer_slice_num):
            size = base + 1 if i < rem else base
            boundaries.append((start, start + size))
            start += size
        return boundaries

    # ------------------------------------------------------------------ #
    # Pick methods (analogous to edge-side PDSeparatedScheduler)         #
    # ------------------------------------------------------------------ #
    def _pick_prefill_batch(self) -> ScheduledBatch:
        """Pick a prefill or prefill-like batch from the ready queues.

        Checks in priority order: active prefill slices (continuation of
        a previously sliced prefill), fresh prefills from ``ready_prefills``,
        then PD-mix batches from ``ready_pdmixes``.

        Caller must ensure at least one source is non-empty before calling.
        """
        if self._active_prefill_slices:
            return self._build_active_prefill_slice_batch()
        if self.ready_prefills:
            return self._build_batch(self.ready_prefills.popleft())
        assert self.ready_pdmixes, (
            "_pick_prefill_batch called with no prefill work available"
        )
        return self._build_batch(self.ready_pdmixes.popleft())

    def _pick_decode_batch(self) -> ScheduledBatch:
        """Pick a decode batch from ``ready_decodes``.

        Caller must ensure ``ready_decodes`` is non-empty before calling.
        """
        return self._build_batch(self.ready_decodes.popleft())

    def _pick_draft_batch(self) -> ScheduledBatch:
        """Pick a draft batch from ``ready_drafts``.

        Caller must ensure ``ready_drafts`` is non-empty before calling.
        """
        return self._build_batch(self.ready_drafts.popleft())

    def _pick_decode_or_draft_by_arrival(self) -> ScheduledBatch:
        """Pick between the head decode and head draft by arrival order.

        DECODE_FIRST and DRAFT_FIRST payloads share the DECODE hidden
        channel, and the edge publishes control messages in exactly the
        order its data plane requires.  Letting a later-arrived draft
        overtake an earlier decode (unconditional draft priority) makes
        the cloud post a recv for the draft payload while the edge's
        next in-flight message is a decode payload of a different size;
        the cloud then never produces the decode response the edge is
        blocked on, and the edge never sends the draft payload the cloud
        is blocked on -- a cross-side deadlock.  Fall back to draft
        priority only when an arrival seq is unavailable.

        Caller must ensure at least one of the two queues is non-empty.
        """
        decode_seq = (
            self._arrival_seq(self.ready_decodes[0])
            if self.ready_decodes
            else None
        )
        draft_seq = (
            self._arrival_seq(self.ready_drafts[0])
            if self.ready_drafts
            else None
        )
        if (
            decode_seq is not None
            and draft_seq is not None
            and decode_seq < draft_seq
        ):
            return self._pick_decode_batch()
        if self.ready_drafts:
            return self._pick_draft_batch()
        return self._pick_decode_batch()

    def _ready_prefill_is_sliced_first_block(self) -> bool:
        if not self.ready_prefills:
            return False
        slices = self._slice_for(self.ready_prefills[0])
        return len(slices) > 1 and isinstance(slices[0], LayerSliceInfo)

    def _ready_prefill_head_is_dummy(self) -> bool:
        """True when the ready_prefills head is an edge placeholder dummy.

        A PREFILL_FIRST dummy carries total_num_scheduled_tokens==0 (the
        is_pd_dummy marker is lost in zmq serialization, so the cloud
        detects it by tokens==0).  It is never sliced, so on a single DP it
        needs no arrival-order arbitration.  But under coordinated + replay
        scheduling, DP0's prefill decision propagates to peer DPs via
        _replay_by_deltas, and the edge-cloud hidden channel requires both
        sides to run the same batch_type per tick.  Letting a dummy skip
        _schedule_by_arrival can therefore force the cloud to run prefill
        while the edge runs decode (the decode arrived first) -- a
        cross-side hidden-channel mismatch deadlock.  Route dummies through
        _schedule_by_arrival so the arrival order still wins.
        """
        if not self.ready_prefills:
            return False
        return self.ready_prefills[0].total_num_scheduled_tokens == 0

    def _schedule_by_arrival(self) -> ScheduledBatch:
        prefill_seq = self._arrival_seq(self.ready_prefills[0])
        decode_seq = self._arrival_seq(self.ready_decodes[0])
        draft_seq = (
            self._arrival_seq(self.ready_drafts[0])
            if self.ready_drafts
            else None
        )
        # Decodes and drafts share the DECODE hidden channel, so the
        # "channel work" competing with prefill slice-0 is whichever of
        # the two arrived first -- an earlier draft must not be
        # overtaken by a later decode either (same deadlock hazard as
        # the reverse, see _pick_decode_or_draft_by_arrival).
        channel_seq = decode_seq
        if draft_seq is not None and (
            channel_seq is None or draft_seq < channel_seq
        ):
            channel_seq = draft_seq
        if prefill_seq is None or channel_seq is None:
            self.cloud_scheduling_state = CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT
            self._start_prefill_middle_throttle()
            return self._build_batch(self.ready_prefills.popleft())
        if channel_seq < prefill_seq:
            logger.info(
                "[PD-PASSIVE] Decode/draft arrived before prefill slice-0: "
                "channel_seq=%d, prefill_seq=%d",
                channel_seq,
                prefill_seq,
            )
            self._clear_prefill_middle_throttle()
            return self._pick_decode_or_draft_by_arrival()
        logger.info(
            "[PD-PASSIVE] Prefill slice-0 arrived before decode/draft: "
            "prefill_seq=%d, channel_seq=%d",
            prefill_seq,
            channel_seq,
        )
        self.cloud_scheduling_state = CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT
        self._start_prefill_middle_throttle()
        return self._build_batch(self.ready_prefills.popleft())

    # ------------------------------------------------------------------ #
    # EXPECT_ALTERNATION: decision / coordination / application           #
    # ------------------------------------------------------------------ #

    def sync_queue_state(self) -> bool:
        """Sync queue lengths across cloud DPs via all_reduce.

        Checks that ready_prefills, ready_decodes, ready_pdmixes,
        _active_prefill_slices, and _active_sliced_prefill have consistent
        lengths/non-None status across all DPs.  If any queue is out of
        sync, the method returns False so the caller can sleep and retry.

        When dp_coord_group is not set or coordination is disabled,
        returns True immediately (no sync needed).
        """
        if self.dp_coord_group is None or not self._is_coordinated_dp():
            return True

        import torch
        import torch.distributed as dist
        _dp_size = getattr(
            self.vllm_config.parallel_config, "data_parallel_size", 1
        )
        _dp_rank = getattr(
            self.vllm_config.parallel_config, "data_parallel_rank", 0
        )

        # 5 fields: ready_prefills, ready_decodes, ready_pdmixes,
        #           _active_prefill_slices, has _active_sliced_prefill
        local = [
            len(self.ready_prefills),
            len(self.ready_decodes),
            len(self.ready_pdmixes),
            len(self._active_prefill_slices),
            1 if self._active_sliced_prefill is not None else 0,
        ]

        # Each rank writes to its own row; SUM leaves values independent.
        tensor = torch.zeros(_dp_size * 5, dtype=torch.int32, device="cpu")
        base = _dp_rank * 5
        for i, v in enumerate(local):
            tensor[base + i] = v
        _cc_t0 = time.monotonic()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM,
                         group=self.dp_coord_group)
        _cc_dt_ms = (time.monotonic() - _cc_t0) * 1000
        # Any work across all cloud DPs? (5 fields/DP: pf/dec/pdmix/slices/active)
        # After this all_reduce both DPs hold the same view, so both make the
        # same skip decision -> barrier pairing preserved.
        has_any_work = any(
            int(tensor[i].item()) != 0 for i in range(_dp_size * 5)
        )
        self._synced_has_any_work = has_any_work
        # 仅在有工作时打印，避免空闲刷屏（空闲时仍保留 1 次 all_reduce 作为探测）
        if has_any_work:
            logger.error(
                "[EC-PERF][CLOUD-COORD] dp_rank=%s phase=sync_queue "
                "all_reduce=%.3fms has_work=%s",
                self.vllm_config.parallel_config.data_parallel_rank,
                _cc_dt_ms, has_any_work,
            )

        # Verify all rank values are identical for each of the 5 queues.
        all_match = True
        for q in range(5):
            vals = [int(tensor[r * 5 + q].item()) for r in range(_dp_size)]
            if len(set(vals)) != 1:
                _names = ["ready_prefills", "ready_decodes", "ready_pdmixes",
                          "_active_prefill_slices", "_active_sliced_prefill"]
                logger.warning(
                    "[SYNC-QUEUE] rank=%s %s mismatch: %s",
                    _dp_rank, _names[q], vals,
                )
                all_match = False
        return all_match

    def _schedule_expect_alternation(self) -> ScheduledBatch:
        """EE 交替调度入口。

        非协调模式：直通 _schedule_expect_alternation_simple。
        协调模式（all_reduce）：
          - DP0 执行 simple，记录前后队列快照。
          - DP0 写入差值 tensor，DP1+ 写零，all_reduce(SUM) 传播。
          - DP1+ 根据差值 _replay_by_deltas 复刻执行。
        """
        if self.dp_coord_group is None or not self._is_coordinated_dp():
            self._coordinated_total_slices = None
            return self._schedule_expect_alternation_simple()

        import torch
        import torch.distributed as dist

        _dp_rank = self.vllm_config.parallel_config.data_parallel_rank

        # Pre-sync the prefill slice *intent* across DPs *before* DP0
        # executes.  Intent = resolve(head tokens) only when this DP has a
        # decode demand (ready_decodes non-empty OR head cloud_suggest), else
        # 0.  all_reduce(MAX) makes both DPs slice when *either* DP has decode
        # demand, so a dummy on DP0 follows DP1's real prefill (which carries
        # the cloud_suggest / decode signal); cold-start (no decode demand on
        # either DP) -> intent 0 -> no slice.  _slice_for honors this on DP0;
        # DP1 follows via d_slices in _replay_by_deltas.
        _has_decode_demand = bool(self.ready_decodes) or (
            bool(self.ready_prefills)
            and getattr(self.ready_prefills[0], "cloud_suggest_slicing", False)
        )
        _local_intent = (
            self._resolve_slice_count(
                self.ready_prefills[0].total_num_scheduled_tokens
            )
            if (self.ready_prefills and _has_decode_demand) else 0
        )
        _sync = torch.tensor([_local_intent], dtype=torch.int32)
        _cc_t0 = time.monotonic()
        dist.all_reduce(_sync, op=dist.ReduceOp.MAX,
                        group=self.dp_coord_group)
        logger.error(
            "[EC-PERF][CLOUD-COORD] dp_rank=%s phase=intent_max all_reduce=%.3fms",
            _dp_rank, (time.monotonic() - _cc_t0) * 1000,
        )
        self._coordinated_total_slices = int(_sync.item()) or None

        if _dp_rank == 0:
            # --- DP0: 执行 + 记录 ---
            before = self._queue_snapshot()
            batch = self._schedule_expect_alternation_simple()
            after = self._queue_snapshot()

            d_prefills = after[0] - before[0]
            d_decodes  = after[1] - before[1]
            d_pdmixes  = after[2] - before[2]
            d_slices   = after[3] - before[3]

            logger.debug(
                "[COORD-SNAPSHOT] DP0 before=%s after=%s "
                "deltas=(pf=%d, dec=%d, dmix=%d, slices=%d)",
                before, after,
                d_prefills, d_decodes, d_pdmixes, d_slices,
            )
        else:
            d_prefills = d_decodes = d_pdmixes = d_slices = 0
            batch = None

        # all_reduce: DP0 写入，DP1+ 写零 → SUM 后所有人拿到 DP0 的值
        _tensor = torch.tensor(
            [d_prefills, d_decodes, d_pdmixes, d_slices],
            dtype=torch.int32,
        )
        _cc_t0 = time.monotonic()
        dist.all_reduce(_tensor, op=dist.ReduceOp.SUM,
                         group=self.dp_coord_group)
        logger.error(
            "[EC-PERF][CLOUD-COORD] dp_rank=%s phase=deltas_sum all_reduce=%.3fms",
            _dp_rank, (time.monotonic() - _cc_t0) * 1000,
        )
        d_prefills, d_decodes, d_pdmixes, d_slices = _tensor.tolist()

        if _dp_rank == 0:
            return batch
        else:
            logger.debug(
                "[COORD-REPLAY] DP%d deltas=(pf=%d, dec=%d, dmix=%d, slices=%d)",
                _dp_rank, d_prefills, d_decodes, d_pdmixes, d_slices,
            )
            return self._replay_by_deltas(
                d_prefills, d_decodes, d_pdmixes, d_slices,
            )

    def _is_coordinated_dp(self) -> bool:
        """True when cross-DP coordination should be active on the cloud side:
        dp_coord_group is set AND model is MoE AND PD-separation is enabled."""
        return (
            bool(getattr(self.vllm_config.model_config, "is_moe", False))
            and getattr(
                self.vllm_config.parallel_config, "enable_edge_cloud", False
            )
        )

    def _schedule_expect_alternation_simple(self) -> ScheduledBatch:
        """Original single-DP EEP/EED state machine (no cross-DP coord)."""
        state = self.cloud_scheduling_state
        logger.debug(
            "[COORD-DIAG] DP%s simple-enter state=%s",
            getattr(self.vllm_config.parallel_config, "data_parallel_rank", 0),
            state.name,
        )
        if state == CloudSchedulingState.EXPECT_EXECUTE_PREFILL:
            if self._active_prefill_slices:
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT
                )
                self._start_prefill_middle_throttle()
                return self._pick_prefill_batch()
            if self.ready_prefills:
                if (
                    self.ready_decodes
                    and (
                        self._ready_prefill_is_sliced_first_block()
                        or self._ready_prefill_head_is_dummy()
                    )
                ):
                    return self._schedule_by_arrival()
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT
                )
                if getattr(
                    self.ready_prefills[0], "cloud_suggest_slicing", False
                ):
                    self._start_prefill_middle_throttle()
                return self._pick_prefill_batch()
            # No Prefill: callback to Decode/Draft.  Arrival order is
            # mandatory here (shared DECODE channel), not a preference.
            if self.ready_drafts or self.ready_decodes:
                self._clear_prefill_middle_throttle()
                return self._pick_decode_or_draft_by_arrival()
        else:  # EXPECT_EXECUTE_DECODE_OR_DRAFT
            # Decode/Draft in arrival order (shared DECODE channel --
            # see _pick_decode_or_draft_by_arrival).
            if self.ready_drafts or self.ready_decodes:
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_PREFILL
                )
                self._clear_prefill_middle_throttle()
                return self._pick_decode_or_draft_by_arrival()
            # No Draft/Decode: callback to Prefill.  Stay in the current
            # state — the next schedule() call will check for drafts
            # again at its earliest opportunity.
            if self._can_fallback_to_prefill_in_decode_state():
                if self._active_prefill_slices:
                    self._start_prefill_middle_throttle()
                    return self._pick_prefill_batch()
                if self.ready_prefills:
                    if getattr(
                        self.ready_prefills[0],
                        "cloud_suggest_slicing", False
                    ):
                        self._start_prefill_middle_throttle()
                    return self._pick_prefill_batch()
            else:
                return ScheduledBatch.empty()

        if self.ready_pdmixes:
            if (
                state == CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT
                and not self._can_fallback_to_prefill_in_decode_state()
            ):
                return ScheduledBatch.empty()
            if state == CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT:
                self._start_prefill_middle_throttle()
            return self._pick_prefill_batch()
        return ScheduledBatch.empty()

    def _queue_snapshot(self) -> tuple[int, int, int, int]:
        """返回 (prefills_len, decodes_len, pdmixes_len, slices_len)。"""
        return (
            len(self.ready_prefills),
            len(self.ready_decodes),
            len(self.ready_pdmixes),
            len(self._active_prefill_slices),
        )

    def _replay_prefill_with_slices(
        self, d_slices: int,
    ) -> ScheduledBatch:
        """复刻 DP0 的 prefill 操作，绕过 _slice_for 的动态决策。

        _slice_for 依赖 self.ready_decodes / cloud_suggest_slicing 等 DP 本地
        队列状态，DP1 复刻时这些状态可能与 DP0 执行时不同，导致切片数量不一致。
        该方法直接使用 d_slices 反推 total_slices，确保 DP1 与 DP0 产生相同的
        _active_prefill_slices 变化。
        """
        so = self.ready_prefills.popleft()
        if d_slices > 0:
            slices = self._do_slice(so, total_slices=d_slices + 1)
            batch = self._gen_batch_by_slices(so, slices)
        else:
            batch = ScheduledBatch(scheduler_output=so, slices=[None])
        self._log_picked_batch(batch)
        return batch

    def _replay_by_deltas(
        self,
        d_prefills: int,
        d_decodes: int,
        d_pdmixes: int,
        d_slices: int,
    ) -> ScheduledBatch:
        """非 DP0：根据 DP0 的队列差值复刻操作。

        total_slices 通过 d_slices 反推（d_slices + 1），无需单独传输。
        """
        if d_slices == -1:
            return self._build_active_prefill_slice_batch()
        elif d_prefills == -1:
            return self._replay_prefill_with_slices(d_slices)
        elif d_decodes == -1:
            return self._build_batch(
                self.ready_decodes.popleft(),
            )
        elif d_pdmixes == -1:
            return self._build_batch(
                self.ready_pdmixes.popleft(),
            )
        else:
            return ScheduledBatch.empty()

    # ------------------------------------------------------------------ #
    # Alternative dispatch helpers                                       #
    # ------------------------------------------------------------------ #

    def _gen_batch_by_slices(
        self, so: SchedulerOutput, slices: list,
    ) -> ScheduledBatch:
        """从 so + slices 列表构建 ScheduledBatch。

        当 len(slices) > 1 时，第一个 slice 作为本次 batch，
        其余转为 SliceTask 追加到 _active_prefill_slices。
        """
        if len(slices) <= 1:
            return ScheduledBatch(scheduler_output=so, slices=slices)

        first_slice = slices[0]
        assert isinstance(first_slice, LayerSliceInfo)
        self._active_sliced_prefill = so
        self._active_prefill_slices.extend(
            SliceTask(so, slice_info)
            for slice_info in slices[1:]
            if isinstance(slice_info, LayerSliceInfo)
        )
        return ScheduledBatch(scheduler_output=so, slices=[first_slice])

    def _schedule_from_queue(self, queue_name: str) -> ScheduledBatch:
        if self._active_prefill_slices:
            if queue_name == "ready_decodes" and self.ready_decodes:
                return self._pick_decode_batch()
            if queue_name in ("ready_prefills", "ready_pdmixes"):
                return self._pick_prefill_batch()
            return ScheduledBatch.empty()

        if queue_name == "ready_prefills":
            if self.ready_prefills:
                return self._build_batch(self.ready_prefills.popleft())
            return ScheduledBatch.empty()
        if queue_name == "ready_decodes":
            if self.ready_decodes:
                return self._pick_decode_batch()
            return ScheduledBatch.empty()
        if queue_name == "ready_drafts":
            if self.ready_drafts:
                return self._pick_draft_batch()
            return ScheduledBatch.empty()
        if queue_name == "ready_pdmixes":
            if self.ready_pdmixes:
                return self._pick_prefill_batch()
            return ScheduledBatch.empty()

        q: deque[SchedulerOutput] = getattr(self, queue_name)
        if q:
            return self._build_batch(q.popleft())
        return ScheduledBatch.empty()

    def _build_batch(
        self, so: SchedulerOutput, total_slices: Optional[int] = None,
    ) -> ScheduledBatch:
        slices = self._slice_for(so, total_slices)
        batch = self._gen_batch_by_slices(so, slices)
        self._log_picked_batch(batch)
        return batch

    def _build_active_prefill_slice_batch(self) -> ScheduledBatch:
        task = self._active_prefill_slices.popleft()
        if not self._active_prefill_slices:
            self._active_sliced_prefill = None
        batch = ScheduledBatch(
            scheduler_output=task.scheduler_output,
            slices=[task.slice_info],
        )
        self._log_picked_batch(batch)
        return batch

    def _log_picked_batch(self, batch: ScheduledBatch) -> None:
        so = batch.scheduler_output
        logger.debug(
            "PassiveScheduler.schedule[%s] picked batch_type=%s slices=%d; "
            "pending=(prefills=%d, active_prefill_slices=%d, "
            "pdmixes=%d, drafts=%d, decodes=%d) seq=%s",
            self.dispatch_policy.value,
            so.batch_type.value if so.batch_type is not None else "<none>",
            len(batch.slices),
            len(self.ready_prefills),
            len(self._active_prefill_slices),
            len(self.ready_pdmixes),
            len(self.ready_drafts),
            len(self.ready_decodes),
            self._arrival_seq(so),
        )

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def has_pending(self) -> bool:
        return bool(
            self.ready_prefills
            or self._active_prefill_slices
            or self.ready_pdmixes
            or self.ready_drafts
            or self.ready_decodes
        )

    @property
    def num_pending(self) -> int:
        return (
            len(self.ready_prefills)
            + len(self._active_prefill_slices)
            + len(self.ready_pdmixes)
            + len(self.ready_drafts)
            + len(self.ready_decodes)
        )
