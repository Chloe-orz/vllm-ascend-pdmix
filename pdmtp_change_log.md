# PD-MTP Change Log

| 序号 | 阶段 | 任务 | 涉及文件 | 状态机变化 |
|---|---|---|---|---|
| 1 | M1 | 将 Qwen-MTP edge-cloud 场景下的 draft 生成从 `VERIFY_LAST.sample_tokens()` 同步路径中拆出，先记录 pending draft 上下文 | `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 2 | M1 | 允许 spec decode 在 draft 结果尚未 ready 时跳过本轮 `request.spec_token_ids` 回填，为后续异步 MTP_DRAFT 任务接入留出口 | `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
| 3 | M2 | 新增 Qwen-MTP draft 的基础 batch 类型，并让 Worker/通信层能识别 `MTP_DRAFT_FIRST/LAST` 执行语义 | `vllm/v1/core/sched/output.py`; `vllm_ascend/core/pd_separated_scheduler.py`; `vllm_ascend/worker/worker.py`; `vllm_ascend/distributed/parallel_state.py`; `vllm_ascend/patch/worker/patch_distributed.py`; `vllm_ascend/scheduler_conflicts.py` | 无 |
| 4 | M2 | 云侧新增 `ready_mtp_drafts` 队列，`MTP_DRAFT_FIRST` 单独分类并按固定优先级选择 | `vllm_ascend/core/passive_scheduler.py` | 云侧：`P中 > DRAFT中 > VERIFY中` |
| 5 | M2 | 在 `SchedulerOutput` 增加 MTP draft 控制面身份字段：父请求、draft task id、draft step index | `vllm/v1/core/sched/output.py`; `vllm_ascend/scheduler_conflicts.py` | 无 |
| 6 | M2 | 边侧 Scheduler 新增 MTP draft 首/尾 ready 队列和 inflight 计数，预留 MTP 调度插槽 | `vllm_ascend/core/pd_separated_scheduler.py` | 边侧：`P首 > P尾 > MTP首 > MTP尾 > VERIFY首 > VERIFY尾` |
| 7 | M2 | EngineCore POST_OUT 支持 `MTP_DRAFT_LAST` 回流到边侧 MTP 尾段队列，并为 `MTP_DRAFT_FIRST` 分配/发布 `head_token` | `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
| 8 | M2 | 迁移 Qwen-MTP 动态 payload 通信入口，新增 MTP 专用 send/recv wrapper 并默认复用 `DECODE` hidden channel | `vllm_ascend/distributed/parallel_state.py` | 无 |
| 9 | M2 | 迁移 Qwen3.5 MTP 模型侧 edge-cloud segment patch，支持 draft 首段/中段/尾段分段执行 | `vllm_ascend/patch/models/qwen3_5_edge_cloud.py` | 无 |
| 10 | M2 | 边云加载流程补充 Qwen-MTP drafter 加载、MTP segment 创建和中间张量稳定 buffer | `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 11 | M2 | 新增云侧单步 Qwen-MTP draft 中段执行方法，接收动态 payload、构造 draft attention metadata 并通过 `DECODE` 数据通道回传 | `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 12 | M2 | 云侧 POST_OUT 控制面增加 `MTP_DRAFT_FIRST -> MTP_DRAFT_LAST` 映射 | `vllm_ascend/v1/engine/passive_core.py` | 无 |
| 13 | M2 | Worker 云侧分发 `MTP_DRAFT_FIRST` 到 Qwen-MTP 单步中段执行路径，避免误用普通 P/D hidden meta | `vllm_ascend/worker/worker.py` | 无 |
| 14 | M2 | 将暂存的 Qwen-MTP pending draft 上下文封装成可由 Worker 取出的 `MTP_DRAFT_FIRST SchedulerOutput` | `vllm_ascend/worker/model_runner_v1.py`; `vllm_ascend/worker/worker.py` | 无 |
| 15 | M2 | 边侧新增 Qwen-MTP 单步 draft 首段/尾段执行入口：首段发送 MTP payload，尾段接收云侧 hidden 并生成本 step draft token | `vllm_ascend/worker/model_runner_v1.py`; `vllm_ascend/worker/worker.py` | 无 |
| 16 | M2 | EngineCore 在 verify 输出更新后拉取 pending MTP draft，并入队到边侧 `mtp_drafts_first_ready` | `vllm/v1/executor/abstract.py`; `vllm/v1/executor/uniproc_executor.py`; `vllm/v1/executor/multiproc_executor.py`; `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
| 17 | M2 | MTP draft 尾段完成后推进下一 draft step，同一 `mtp_draft_task_id` 递增 `draft_step_idx` 重新入队直到达到 `num_spec_tokens` | `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 18 | M2 | 最后一个 MTP draft step 完成后组装 `DraftTokenIds` 并通过父 verify `SchedulerOutput` 回填 Scheduler | `vllm/v1/executor/abstract.py`; `vllm/v1/executor/uniproc_executor.py`; `vllm/v1/executor/multiproc_executor.py`; `vllm_ascend/worker/model_runner_v1.py`; `vllm_ascend/worker/worker.py`; `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
| 19 | M2 | 收敛数据面设计：取消独立 `MTP_DRAFT` hidden channel，MTP draft 复用 `DECODE` channel，并按原 Decode 首段发送语义增加 MTP/VERIFY 首段互斥 | `vllm/v1/core/sched/output.py`; `vllm_ascend/core/pd_separated_scheduler.py`; `vllm_ascend/distributed/parallel_state.py`; `vllm_ascend/patch/worker/patch_distributed.py`; `vllm_ascend/worker/worker.py`; `vllm_ascend/worker/model_runner_v1.py`; `vllm_ascend/scheduler_conflicts.py` | 无 |
| 20 | M2 | 增加 MTP draft 远端等待计数，阻止 `MTP_DRAFT_FIRST` 已发送但 `MTP_DRAFT_LAST` 未完成时提前调度 VERIFY | `vllm_ascend/core/pd_separated_scheduler.py` | strict MTP：`MTP首 -> 等 MTP尾 -> 下一步 MTP首或 VERIFY首` |
| 21 | M2 | 非 batch_queue 路径下允许 `MTP_DRAFT_FIRST` 直接发布 PRE_OUT 控制面，避免云侧收不到 DRAFT中任务 | `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
| 22 | M2 | 请求结束时清理 stale MTP draft 首/尾队列，丢弃已回包未执行的 MTP尾时同步释放远端等待计数 | `vllm_ascend/core/pd_separated_scheduler.py` | 无 |
| 23 | M2 | 请求结束/abort 后通过 Executor RPC 清理 Worker 侧 pending MTP draft context，避免已结束请求重新入队 draft | `vllm/v1/executor/abstract.py`; `vllm/v1/executor/uniproc_executor.py`; `vllm/v1/executor/multiproc_executor.py`; `vllm_ascend/patch/platform/patch_engine_core.py`; `vllm_ascend/worker/worker.py`; `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 24 | M2 | MTP draft 数据面 payload 增加 `head_token`、`mtp_draft_task_id`、`draft_step_idx` 身份字段，并在云侧/边侧尾段校验 task 与 step 顺序 | `vllm_ascend/worker/worker.py`; `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 25 | M2 | MTP 初始化时校验 `num_spec_tokens` 和 `num_mtp_layers` 均为正，并打印二者关系；不要求二者相等 | `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 26 | M2 | 统一普通 step 与 batch_queue 路径的 MTP/P/D 首段 `head_token` 分配，并校验 MTP_DRAFT_LAST 控制面身份字段 | `vllm_ascend/patch/platform/patch_engine_core.py`; `vllm_ascend/v1/engine/passive_core.py`; `vllm_ascend/core/pd_separated_scheduler.py` | 无 |
| 27 | M2 | 将 MTP draft 首/尾队列和远端等待状态纳入 `has_requests()`，避免严格 MTP 派生任务被 EngineCore 误判为空闲 | `vllm_ascend/core/pd_separated_scheduler.py` | 无 |
| 28 | M2 | 对齐主仓 Executor 与 Ascend Worker/ModelRunner 的 MTP draft 清理 RPC 参数类型，允许 `set[str]` 直接透传 | `vllm_ascend/worker/worker.py`; `vllm_ascend/worker/model_runner_v1.py` | 无 |
| 29 | M2 | 在 Qwen-MTP 边云拆分模式下跳过 batch_queue deferred 分支的旧同步 `take_draft_token_ids()` 回填，只使用 MTP_DRAFT 完成结果回填 | `vllm_ascend/patch/platform/patch_engine_core.py` | 无 |
| 30 | M2 | 已结束请求的远端 `MTP_DRAFT_LAST` 回包在调度器侧识别为 stale 并丢弃，避免进入 ModelRunner 后找不到 pending draft context | `vllm_ascend/core/pd_separated_scheduler.py` | 无 |
| 31 | M2 | 将 Qwen-MTP pending draft 从单槽位改为 `task_id -> context` 字典加 pending task 队列，支持多个 P尾/VERIFY尾 派生 draft 并存 | `vllm_ascend/worker/model_runner_v1.py` | 无 |
