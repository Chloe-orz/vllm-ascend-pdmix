# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the minimal edge-cloud concurrency backport."""

from collections import deque
from unittest.mock import MagicMock

from vllm.v1.core.sched.output import BatchType, HiddenChannelType


def _scheduler_output(batch_type: BatchType, req_ids: tuple[str, ...]):
    output = MagicMock()
    output.batch_type = batch_type
    output.draft_task_id = "task-0"
    output.draft_step_idx = 0
    output.hidden_channel = HiddenChannelType.DECODE
    output.parent_req_id = req_ids[0] if req_ids else None
    output.num_scheduled_tokens = {req_id: 1 for req_id in req_ids}
    return output


def _bare_passive_scheduler():
    from vllm_ascend.core.passive_scheduler import PassiveScheduler

    scheduler = PassiveScheduler.__new__(PassiveScheduler)
    scheduler.ready_decodes = deque()
    scheduler.ready_drafts = deque()
    scheduler._build_batch = MagicMock(side_effect=lambda output: output)
    return scheduler


def test_cloud_shared_channel_preserves_decode_before_draft_arrival():
    scheduler = _bare_passive_scheduler()
    decode = _scheduler_output(BatchType.DECODE_FIRST, ("req-d",))
    draft = _scheduler_output(BatchType.DRAFT_FIRST, ("req-r",))
    scheduler._remember_arrival_seq(decode, 10)
    scheduler._remember_arrival_seq(draft, 11)
    scheduler.ready_decodes.append(decode)
    scheduler.ready_drafts.append(draft)

    assert scheduler._pick_decode_or_draft_by_arrival() is decode


def test_cloud_shared_channel_preserves_draft_before_decode_arrival():
    scheduler = _bare_passive_scheduler()
    decode = _scheduler_output(BatchType.DECODE_FIRST, ("req-d",))
    draft = _scheduler_output(BatchType.DRAFT_FIRST, ("req-r",))
    scheduler._remember_arrival_seq(draft, 10)
    scheduler._remember_arrival_seq(decode, 11)
    scheduler.ready_decodes.append(decode)
    scheduler.ready_drafts.append(draft)

    assert scheduler._pick_decode_or_draft_by_arrival() is draft


def _bare_edge_scheduler():
    from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

    scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
    scheduler.drafts_first_ready = deque()
    scheduler.drafts_last_ready = deque()
    scheduler.requests = {}
    scheduler.draft_remote_pending_count = 1
    scheduler._validate_draft_tail_channel = MagicMock()
    scheduler._make_empty_batch = MagicMock(return_value="EMPTY")
    return scheduler


def test_partial_finish_keeps_batch_scoped_draft_head():
    scheduler = _bare_edge_scheduler()
    output = _scheduler_output(
        BatchType.DRAFT_FIRST, ("req-finished", "req-live")
    )
    scheduler.requests["req-live"] = MagicMock()
    scheduler.drafts_first_ready.append(output)

    scheduler._drop_stale_drafts_for_req_ids({"req-finished"})

    assert list(scheduler.drafts_first_ready) == [output]


def test_stale_draft_tail_is_returned_for_channel_drain():
    scheduler = _bare_edge_scheduler()
    output = _scheduler_output(BatchType.DRAFT_LAST, ("req-finished",))
    scheduler.drafts_last_ready.append(output)

    assert scheduler._pick_draft_last_batch() is output
    assert scheduler.draft_remote_pending_count == 1


def test_worker_keeps_context_until_all_batch_requests_finish():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._pending_edge_cloud_draft_contexts = {
        "task-0": {"req_ids": ("req-0", "req-1")}
    }
    runner._pending_edge_cloud_draft_task_ids = deque(["task-0"])

    runner.clear_pending_edge_cloud_draft_for_req_ids({"req-0"})
    assert "task-0" in runner._pending_edge_cloud_draft_contexts

    runner.clear_pending_edge_cloud_draft_for_req_ids({"req-1"})
    assert "task-0" not in runner._pending_edge_cloud_draft_contexts
    assert not runner._pending_edge_cloud_draft_task_ids
