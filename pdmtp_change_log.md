# PD-MTP Change Log

| 序号 | 阶段 | 任务 | 涉及文件 | 状态机变化 |
|---|---|---|---|---|
| 1 | M1 | 将 Qwen-MTP edge-cloud 场景下的 draft 生成从 `VERIFY_LAST.sample_tokens()` 同步路径中拆出，先记录 pending draft 上下文 | `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 2 | M1 | 允许 spec decode 在 draft 结果尚未 ready 时跳过本轮 `request.spec_token_ids` 回填，为后续异步 MTP_DRAFT 任务接入留出口 | `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
| 3 | M2 | 新增 Qwen-MTP draft 的基础 batch 类型和独立 hidden channel，并让 Worker/通信层能识别 `MTP_DRAFT` 通道 | `vllm/v1/core/sched/output.py`; `vllm_ascend/core/pd_separated_scheduler.py`; `vllm_ascend/worker/worker.py`; `vllm_ascend/distributed/parallel_state.py`; `vllm_ascend/patch/worker/patch_distributed.py`; `vllm_ascend/scheduler_conflicts.py` | 无 |
| 4 | M2 | 云侧新增 `ready_mtp_drafts` 队列，`MTP_DRAFT_FIRST` 单独分类并按固定优先级选择 | `vllm_ascend/core/passive_scheduler.py` | 云侧：`P中 > DRAFT中 > VERIFY中` |
| 5 | M2 | 在 `SchedulerOutput` 增加 MTP draft 控制面身份字段：父请求、draft task id、draft step index | `vllm/v1/core/sched/output.py`; `vllm_ascend/scheduler_conflicts.py` | 无 |
| 6 | M2 | 边侧 Scheduler 新增 MTP draft 首/尾 ready 队列和 inflight 计数，预留 MTP 调度插槽 | `vllm_ascend/core/pd_separated_scheduler.py` | 边侧：`P首 > P尾 > MTP首 > MTP尾 > VERIFY首 > VERIFY尾` |
| 7 | M2 | EngineCore POST_OUT 支持 `MTP_DRAFT_LAST` 回流到边侧 MTP 尾段队列，并为 `MTP_DRAFT_FIRST` 分配/发布 `head_token` | `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
| 8 | M2 | 迁移 Qwen-MTP 动态 payload 通信入口，新增 MTP 专用 send/recv wrapper 并绑定 `MTP_DRAFT` hidden channel | `vllm_ascend/distributed/parallel_state.py` | 无 |
