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
| 9 | M2 | 迁移 Qwen3.5 MTP 模型侧 edge-cloud segment patch，支持 draft 首段/中段/尾段分段执行 | `vllm_ascend/patch/models/qwen3_5_edge_cloud.py` | 无 |
| 10 | M2 | 边云加载流程补充 Qwen-MTP drafter 加载、MTP segment 创建和中间张量稳定 buffer | `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 11 | M2 | 新增云侧单步 Qwen-MTP draft 中段执行方法，接收动态 payload、构造 draft attention metadata 并通过 `MTP_DRAFT` 通道回传 | `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 12 | M2 | 云侧 POST_OUT 控制面增加 `MTP_DRAFT_FIRST -> MTP_DRAFT_LAST` 映射 | `vllm_ascend/v1/engine/passive_core.py` | 无 |
| 13 | M2 | Worker 云侧分发 `MTP_DRAFT_FIRST` 到 Qwen-MTP 单步中段执行路径，避免误用普通 P/D hidden meta | `vllm_ascend/worker/worker.py` | 无 |
| 14 | M2 | 将暂存的 Qwen-MTP pending draft 上下文封装成可由 Worker 取出的 `MTP_DRAFT_FIRST SchedulerOutput` | `vllm_ascend/worker/model_runner_v1.py`; `vllm_ascend/worker/worker.py` | 无 |
| 15 | M2 | 边侧新增 Qwen-MTP 单步 draft 首段/尾段执行入口：首段发送 MTP payload，尾段接收云侧 hidden 并生成本 step draft token | `vllm_ascend/worker/model_runner_v1.py`; `vllm_ascend/worker/worker.py` | 无 |
| 16 | M2 | EngineCore 在 verify 输出更新后拉取 pending MTP draft，并入队到边侧 `mtp_drafts_first_ready` | `vllm/v1/executor/abstract.py`; `vllm/v1/executor/uniproc_executor.py`; `vllm/v1/executor/multiproc_executor.py`; `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
| 17 | M2 | MTP draft 尾段完成后推进下一 draft step，同一 `mtp_draft_task_id` 递增 `draft_step_idx` 重新入队直到达到 `num_spec_tokens` | `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 18 | M2 | 最后一个 MTP draft step 完成后组装 `DraftTokenIds` 并通过父 verify `SchedulerOutput` 回填 Scheduler | `vllm/v1/executor/abstract.py`; `vllm/v1/executor/uniproc_executor.py`; `vllm/v1/executor/multiproc_executor.py`; `vllm_ascend/worker/model_runner_v1.py`; `vllm_ascend/worker/worker.py`; `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
