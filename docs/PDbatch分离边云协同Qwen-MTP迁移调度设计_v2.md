# PDbatch 分离边云协同 Qwen-MTP 迁移调度设计文档 v2

本文档聚焦 `vllm-ascend` 中 Qwen-MTP 的已有实现，以及该实现如何接入 PDMIX 边云 PD 掩盖调度框架。

相比旧版文档，本版把通信链路放到调度策略之前，先明确：

```text
控制面：SchedulerOutput，通过 PRE_OUT / POST_OUT ZMQ 传输
数据面：IntermediateTensors，通过 PP/HCCL isend/irecv 传输
配对键：SchedulerOutput.head_token == IntermediateTensors["_head_token"]
```

这样可以避免只设计队列和优先级，而遗漏 Qwen-MTP draft 跨边云执行时的通信、配对、挂起态和回填时序。

参考：

- [PDbatch分离分布式边云协同推理设计说明书.md](PDbatch分离分布式边云协同推理设计说明书.md)
- [PDbatch分离分布式边云协同推理调度算法设计.md](PDbatch分离分布式边云协同推理调度算法设计.md)
- [PDbatch分离边云协同Phase2-3详细设计.md](PDbatch分离边云协同Phase2-3详细设计.md)
- [PDbatch分离边云协同Phase2-4 extend详细设计.md](PDbatch分离边云协同Phase2-4%20extend详细设计.md)
- [PDbatch分离边云协同Phase4详细设计.md](PDbatch分离边云协同Phase4详细设计.md)
- [PDbatch分离边云协同Phase5&7详细设计.md](PDbatch分离边云协同Phase5&7详细设计.md)

---

# 1. 设计范围与目标

## 1.1 设计范围

本文档关注：

1. PDMIX 调度框架支持 Qwen-MTP 开启后的 decode verify + draft 生成闭环。
2. `VERIFY_DECODE_LAST` 不再同步阻塞等待 Qwen-MTP draft 生成。
3. Qwen-MTP draft 作为独立派生任务进入边云调度框架。
4. draft 结果通过 `take_draft_token_ids()` 或等价路径回填 Scheduler。
5. 下一轮 `VERIFY_DECODE` 消费 `scheduled_spec_decode_tokens`。
6. Qwen-MTP draft 跨边云执行时，明确控制面、数据面、head_token 配对和挂起态恢复协议。

核心问题不是新增一套独立 decode 主流程，而是把 Qwen-MTP 的 draft/verify 生命周期接进现有 PDMIX PD 掩盖调度。

## 1.2 Qwen-MTP 接入目标

Qwen-MTP 开启后，decode 不应理解为“普通 decode”和“MTP decode”两套流程切换。主请求仍然走 Scheduler 的 decode 调度，只是 decode 请求可能携带上一轮 draft 生成的 `scheduled_spec_decode_tokens`。

因此需要把 decode 明确拆成两条语义链：

| 链路 | 含义 | 用户可见输出 |
|---|---|---|
| `VERIFY_DECODE` | 主模型验证上一轮 draft tokens，并完成 accept/reject | 有 |
| `MTP_DRAFT` | Qwen-MTP proposer 生成下一轮 draft token ids | 无 |

关键点：

```text
MTP_DRAFT 是下一轮 VERIFY_DECODE 的前置准备，不是普通 filler。
```

## 1.3 与现有 PDMIX PD 框架的关系

现有 PDMIX 已经把主模型执行拆成：

```text
P_FIRST(edge) -> P_MIDDLE(cloud) -> P_LAST(edge)
D_FIRST(edge) -> D_MIDDLE(cloud) -> D_LAST(edge)
```

Qwen-MTP 接入后，在此基础上新增派生链：

```text
MTP_DRAFT_FIRST(edge) -> MTP_DRAFT_MIDDLE(cloud) -> MTP_DRAFT_LAST(edge)
```

主模型 verify 继续复用现有 `DECODE_FIRST / DECODE_LAST` 链路，其语义在 MTP 模式下视为：

```text
DECODE_FIRST = VERIFY_DECODE_FIRST
DECODE_LAST  = VERIFY_DECODE_LAST
```

---

# 2. vllm-ascend Qwen-MTP 来源实现分析

## 2.1 关键文件

| 文件 | 作用 |
|---|---|
| `vllm_ascend/spec_decode/llm_base_proposer.py` | Qwen-MTP proposer 主逻辑，包含 `_run_mtp_edge_cloud()`、多 draft step 循环、draft token 计算 |
| `vllm_ascend/patch/worker/patch_qwen3_next_mtp.py` | Qwen3 Next MTP 相关 patch，主要处理 KV cache 绑定兼容 |
| `vllm_ascend/patch/worker/patch_qwen3_dflash.py` | DFlash Qwen3 相关 patch，处理 context KV 预计算 |
| `vllm_ascend/worker/model_runner_v1.py` | 当前 `sample_tokens()` 同步触发 `propose_draft_token_ids()` 的入口 |

## 2.2 Qwen-MTP proposer 调用链

来源实现中，verify 后的 draft 生成大致在以下路径中同步完成：

```text
NPUModelRunner.sample_tokens()
  -> _sample() / accept-reject
  -> propose_draft_token_ids()
      -> Ascend proposer
      -> Qwen-MTP draft model input 构造
      -> _run_mtp_edge_cloud()
      -> draft token ids 生成
  -> _copy_draft_token_ids_to_cpu()
  -> return ModelRunnerOutput
```

这个组织方式在非 PD 掩盖场景中逻辑简单，因为同一个请求的下一轮 verify 本来就依赖 draft。

在 PDMIX 中，`sample_tokens()` 处在边侧尾段处理路径上。如果 draft 继续同步接在 `VERIFY_LAST` 后面，就会推迟其他请求已经 ready 的 `P_LAST / VERIFY_LAST`，影响全局边云流水线。

## 2.3 Qwen-MTP edge/cloud 三段职责

来源实现中，Qwen-MTP draft 的边云切分可以抽象为：

```text
MTP_DRAFT_FIRST(edge)
    -> embed + fc / draft head segment
    -> send hidden + positions + spec_step_idx to cloud

MTP_DRAFT_MIDDLE(cloud)
    -> build MTP draft attention metadata
    -> run Qwen-MTP draft decoder segment
    -> send hidden back to edge

MTP_DRAFT_LAST(edge)
    -> recv cloud hidden
    -> norm / lm_head / draft token generation
    -> produce draft_token_ids
```

## 2.4 `num_speculative_tokens` 与 draft step

`num_speculative_tokens` 来自 `speculative_config.num_speculative_tokens`，不是写死值。

当 `num_speculative_tokens = 3` 时，同一个请求的一轮 draft 是串行的：

```text
draft step0 -> draft step1 -> draft step2
```

后一个 draft token 依赖前一个 draft token，因此不能把同一请求的 3 个 draft step 简单并发发出。

---

# 3. PDMIX 现有 PD 分离基础契约

## 3.1 边云角色与 rank 布局

PDMIX 边云模式中：

| 角色 | 职责 |
|---|---|
| Edge / rank0 leader | Embedding、首段、尾段、LM Head、采样、对外输出 |
| Cloud / passive rank | 中间层计算 |

边侧和云侧可以 TP 不均等。PP 通信组只包含边侧 leader 和云侧 leader：

```text
PP group = [edge_rank0, cloud_rank0]
```

这意味着 Qwen-MTP 的跨边云 hidden state 数据面也必须明确：

```text
谁负责发起 PP isend
谁负责 irecv
非 leader TP rank 如何参与本地 TP 计算
输出是否仍统一回到 edge rank0
```

## 3.2 模型层切分

现有边云模式中：

```text
edge: segment_a + segment_e
cloud: segment_c
```

Qwen-MTP draft 也必须落到同样的执行模型中：

```text
MTP_DRAFT_FIRST  -> edge MTP segment_a
MTP_DRAFT_MIDDLE -> cloud MTP segment_c
MTP_DRAFT_LAST   -> edge MTP segment_e
```

## 3.3 控制面：PRE_OUT / POST_OUT

现有 P/D 链路使用两条 ZMQ 控制面通道：

| 通道 | 方向 | 传输内容 |
|---|---|---|
| PRE_OUT | Edge -> Cloud | `PREFILL_FIRST / DECODE_FIRST` 的 `SchedulerOutput` |
| POST_OUT | Cloud -> Edge | `PREFILL_LAST / DECODE_LAST` 的 `SchedulerOutput` |

MTP_DRAFT 也需要纳入该控制面：

| 通道 | MTP_DRAFT 行为 |
|---|---|
| PRE_OUT | Edge 发送 `MTP_DRAFT_FIRST` |
| POST_OUT | Cloud 返回 `MTP_DRAFT_LAST` |

## 3.4 数据面：PP/HCCL `IntermediateTensors`

现有 P/D 链路的数据面通过 PP/HCCL 传 `IntermediateTensors`：

```text
edge segment_a -> isend hidden -> cloud segment_c
cloud segment_c -> isend hidden -> edge segment_e
```

MTP_DRAFT 数据面同构：

```text
edge MTP_DRAFT_FIRST -> isend draft hidden -> cloud MTP_DRAFT_MIDDLE
cloud MTP_DRAFT_MIDDLE -> isend draft hidden -> edge MTP_DRAFT_LAST
```

## 3.5 `head_token` 与 `_head_token`

PDMIX 已有设计要求：

```text
SchedulerOutput.head_token == IntermediateTensors["_head_token"]
```

原因是控制面 ZMQ 和数据面 PP/HCCL 是两条独立通道，可能出现跨通道乱序。

MTP_DRAFT 必须沿用该协议：

```text
MTP_DRAFT_FIRST SchedulerOutput.head_token
==
MTP_DRAFT hidden tensors["_head_token"]
==
MTP_DRAFT_LAST SchedulerOutput.head_token
```

## 3.6 HeadState 挂起 / 恢复机制

现有 P/D 首尾拆分依赖：

```text
P/D 首段：suspend HeadState
P/D 尾段：resume HeadState
```

Qwen-MTP draft 也需要独立的挂起态：

```text
MTP_DRAFT_FIRST：suspend MTPDraftState
MTP_DRAFT_LAST：resume MTPDraftState
```

不能复用普通 `HeadState`，因为 MTP_DRAFT 不产生用户 token，且要保存 draft step、proposer 输入、target hidden states 等不同字段。

---

# 4. Qwen-MTP 在 PD 中的核心拆分

## 4.1 为什么不能继续把 draft 同步接在 verify 后

原 vllm-ascend 的同步路径可抽象为：

```text
VERIFY_LAST.sample_tokens()
    -> 主模型 logits / sampler / accept-reject
    -> propose_draft_token_ids()
        -> Qwen-MTP FIRST(edge)
        -> Qwen-MTP MIDDLE(cloud)
        -> Qwen-MTP LAST(edge)
    -> take_draft_token_ids()
```

在 PDMIX 中，如果继续把 Qwen-MTP draft 塞在 `VERIFY_LAST.sample_tokens()` 内，会产生两个问题：

1. 当前请求的 `VERIFY_LAST` 返回被 draft 全流程卡住。
2. 同一个边侧 worker 上其他请求的 `P_LAST / VERIFY_LAST` 尾段处理也会被推迟。

迁移后的目标是：

```text
VERIFY_LAST 只负责产出用户 token 和请求状态；
若请求未结束，则创建 MTP_DRAFT task；
MTP_DRAFT task 后续由调度器按优先级执行。
```

## 4.2 同请求内依赖

对同一个 Qwen-MTP 请求：

```text
draft 未 ready -> 下一轮 VERIFY_DECODE_FIRST 不可调度
draft ready    -> 下一轮 VERIFY_DECODE_FIRST 可调度 width=1+N
```

这不是“普通 decode 和 MTP decode 两套流程”，而是 MTP 模式下 verify 的前置条件。

## 4.3 跨请求间调度

拆开 draft 和 verify 的目的不是让同一个请求的 draft 与 verify 并行，而是让调度器能在全局维度显式排序：

```text
reqA 的 MTP_DRAFT
reqB 的 VERIFY_LAST
reqC 的 P_LAST
reqD 的 VERIFY_FIRST
```

当前策略倾向：

```text
P 类保持原 PD 状态机优先；
非 P 类中，MTP_DRAFT 类优先于 VERIFY_DECODE 类；
MTP_DRAFT 内部先首段后尾段。
```

---

# 5. Qwen-MTP 请求生命周期

## 5.1 Prefill 到首轮 draft

Prefill 结束后，边侧 `P_LAST` 产出第一个真实 token。此时还没有上一轮 draft 可 verify，因此需要先生成下一轮 draft：

```text
P_FIRST
  -> P_MIDDLE
  -> P_LAST
      -> 产出第一个真实 token
      -> 创建 MTP_DRAFT(round=1)

MTP_DRAFT(round=1, step=0)_FIRST
  -> MTP_DRAFT(round=1, step=0)_MIDDLE
  -> MTP_DRAFT(round=1, step=0)_LAST
      -> draft_token_1

MTP_DRAFT(round=1, step=1)_FIRST
  -> MTP_DRAFT(round=1, step=1)_MIDDLE
  -> MTP_DRAFT(round=1, step=1)_LAST
      -> draft_token_2

MTP_DRAFT(round=1, step=2)_FIRST
  -> MTP_DRAFT(round=1, step=2)_MIDDLE
  -> MTP_DRAFT(round=1, step=2)_LAST
      -> draft_token_3

Scheduler.update_draft_token_ids([draft_token_1, draft_token_2, draft_token_3])
```

## 5.2 稳定 decode 轮次

下一轮主模型 verify：

```text
VERIFY_DECODE(round=1, width=1+3)_FIRST
  -> VERIFY_DECODE(round=1, width=1+3)_MIDDLE
  -> VERIFY_DECODE(round=1, width=1+3)_LAST
      -> accept/reject
      -> 输出用户可见 token
      -> 若请求未结束，创建 MTP_DRAFT(round=2)
```

稳定形态是：

```text
DRAFT_1 -> DRAFT_2 -> DRAFT_3 -> VERIFY(width=1+3)
```

## 5.3 finished / abort / stale draft

MTP_DRAFT 不改变请求 finished 状态。请求是否结束只由：

```text
P_LAST
VERIFY_DECODE_LAST
```

决定。

如果请求在 MTP_DRAFT 执行期间被 abort：

```text
丢弃对应 MTPDraftTask
清理 MTPDraftState
释放 mtp_draft_inflight_count
忽略已经返回的 draft token ids
```

MTP_DRAFT_LAST 回来后必须校验：

```text
request 仍存在
request 未 abort
draft_task_id / draft_step_idx 匹配
head_token 匹配
```

---

# 6. BatchType 与 SchedulerOutput 扩展

## 6.1 新增 BatchType

推荐最小新增：

```python
BatchType.MTP_DRAFT_FIRST
BatchType.MTP_DRAFT_LAST
```

主模型 verify 继续兼容使用：

```python
BatchType.DECODE_FIRST   # 语义上视为 VERIFY_DECODE_FIRST
BatchType.DECODE_LAST    # 语义上视为 VERIFY_DECODE_LAST
```

draft 输出不是用户 token，必须新增独立 batch type，不能复用 `DECODE_LAST`。

## 6.2 SchedulerOutput 新增字段

`MTP_DRAFT_FIRST` 需要携带：

```python
head_token: str
parent_head_token: str | None
draft_task_id: str
draft_step_idx: int
hidden_channel: HiddenChannelType.MTP_DRAFT
```

字段含义：

| 字段 | 含义 |
|---|---|
| `head_token` | MTP draft 首尾配对 token |
| `parent_head_token` | 触发该 draft 的 `P_LAST` 或 `VERIFY_DECODE_LAST` token |
| `draft_task_id` | draft task 唯一 id |
| `draft_step_idx` | `0..num_speculative_tokens-1` |
| `hidden_channel` | 独立 MTP draft 数据通道 |

## 6.3 云侧 batch 分类

云侧只允许以下 first 类 batch 入 ready queue：

| 入站 batch type | 云侧队列 | 云侧执行 |
|---|---|---|
| `PREFILL_FIRST` | `ready_prefills[]` | P中 |
| `DECODE_FIRST` | `ready_decodes[]` | VERIFY_D中 |
| `MTP_DRAFT_FIRST` | `ready_mtp_drafts[]` | MTP_DRAFT中 |

云侧收到以下 batch 必须丢弃并记录日志：

```text
PREFILL_LAST
DECODE_LAST
MTP_DRAFT_LAST
EMPTY
```

---

# 7. MTP_DRAFT 通信链路设计

## 7.1 控制面链路

MTP_DRAFT 控制面沿用 PRE_OUT / POST_OUT：

```text
Edge Scheduler -> PRE_OUT -> Cloud PassiveScheduler
Cloud -> POST_OUT -> Edge Scheduler
```

映射关系：

| 阶段 | 控制面动作 |
|---|---|
| `MTP_DRAFT_FIRST` | Edge 通过 PRE_OUT 发送给 Cloud |
| `MTP_DRAFT_MIDDLE` | Cloud 从 `ready_mtp_drafts[]` 取出执行 |
| `MTP_DRAFT_LAST` | Cloud 通过 POST_OUT 回传给 Edge |

## 7.2 数据面链路

MTP_DRAFT 数据面通过 PP/HCCL 传 `IntermediateTensors`：

```text
Edge MTP_DRAFT_FIRST
  -> isend(draft hidden + _head_token + positions + spec_step_idx)
Cloud MTP_DRAFT_MIDDLE
  -> irecv(...)
  -> run Qwen-MTP middle
  -> isend(draft hidden + _head_token + spec_step_idx)
Edge MTP_DRAFT_LAST
  -> irecv(...)
  -> resume MTPDraftState
  -> produce draft token ids
```

## 7.3 `head_token` 配对

MTP_DRAFT 必须满足：

```text
MTP_DRAFT_FIRST SchedulerOutput.head_token
==
IntermediateTensors["_head_token"]
==
MTP_DRAFT_LAST SchedulerOutput.head_token
```

边侧 `MTP_DRAFT_LAST` 执行前必须校验：

```python
assert scheduler_output.head_token == intermediate_tensors["_head_token"]
```

若不一致，说明 ZMQ 控制面和 PP 数据面错配，必须报错，不能静默继续。

## 7.4 POST_OUT 发送时机

P/D 现有链路倾向“控制面先行”：云侧可以先 POST_OUT 回传 last 控制面，边侧真正执行 last 时再等待数据面 hidden。

MTP_DRAFT 需要更谨慎，因为本设计把 `MTP_DRAFT_LAST` 优先级放在 verify 前面。如果 `MTP_DRAFT_LAST` 控制面很早到达但 hidden 还没准备好，边侧执行 MTP_DRAFT_LAST 时会阻塞在接收 hidden。

推荐策略：

```text
MTP_DRAFT_LAST 的 POST_OUT 在云侧 MTP_DRAFT_MIDDLE 完成并发起 hidden 回传后发布。
```

这样边侧看到 `mtp_drafts_last_ready[]` 时，数据面更可能已经可接收，减少高优先级 MTP_DRAFT_LAST 阻塞边侧 worker 的风险。

如果实现上暂时复用 P/D 的控制面先行策略，则必须在验证中观察：

```text
MTP_DRAFT_LAST 入口等待 hidden 的时间
MTP_DRAFT_LAST 队列长度
边侧 P_LAST / VERIFY_LAST 是否被 MTP_DRAFT_LAST 阻塞
```

## 7.5 `recv_object` metadata 阻塞风险

`irecv_tensor_dict` 的 tensor body 是异步的，但 metadata 接收有 blocking `recv_object` 握手。Decode 小包路径中这个开销占比很高。

Qwen-MTP 当 `num_speculative_tokens=3` 时，每轮 draft 至少有 3 次小包往返：

```text
step0 edge->cloud + cloud->edge
step1 edge->cloud + cloud->edge
step2 edge->cloud + cloud->edge
```

因此需要把以下指标纳入验证：

```text
MTP_DRAFT edge->cloud metadata RTT
MTP_DRAFT cloud->edge metadata RTT
MTP_DRAFT_LAST 等待 hidden 的时间
num_speculative_tokens=3 时单轮 draft 总通信耗时
```

## 7.6 通信组与 channel 隔离

建议新增：

```python
HiddenChannelType.MTP_DRAFT
```

不建议无保护复用 `HiddenChannelType.DECODE`，原因：

1. `DECODE` channel 已服务主模型 verify。
2. Qwen-MTP draft 有独立首尾生命周期。
3. 复用容易导致 `head_token` 数据面错配。
4. 复用会让 draft send/recv handle 反向阻塞 verify 尾段处理。

首版可以复用同一个 PP group，但需要独立：

```text
send handle 管理
head_token namespace
MTPDraftState dict
hidden channel 标识
日志和统计字段
```

---

# 8. MTPDraftState 设计

## 8.1 为什么需要独立于 HeadState

`HeadState` 表示主模型 P/D 首尾拆分的挂起态。MTP_DRAFT 不产生用户 token，且需要保存 draft step、proposer 输入和目标模型 hidden states，因此需要独立状态：

```python
self._pending_mtp_draft_states: dict[str, MTPDraftState]
```

key 为：

```text
MTP_DRAFT SchedulerOutput.head_token
```

## 8.2 MTPDraftTask

`P_LAST` 或 `VERIFY_DECODE_LAST` 完成后创建：

```python
@dataclass
class MTPDraftTask:
    task_id: str
    req_ids: list[str]
    parent_head_token: str | None
    draft_step_idx: int
    num_speculative_tokens: int
    sampled_token_ids: Any
    target_hidden_states: Any
    positions: Any
    spec_decode_metadata: Any
    spec_decode_common_attn_metadata: Any
    sampling_metadata: Any
    batch_desc: Any
```

## 8.3 MTPDraftState

`MTP_DRAFT_FIRST` 执行完 edge first 后挂起：

```python
@dataclass
class MTPDraftState:
    head_token: str
    task_id: str
    req_ids: list[str]
    draft_step_idx: int
    num_speculative_tokens: int
    parent_head_token: str | None
    positions: Any
    attn_metadata: Any
    sampling_metadata: Any
    batch_desc: Any
    target_hidden_states: Any
    sampled_token_ids: Any
```

## 8.4 suspend / resume

```text
MTP_DRAFT_FIRST:
  build model input
  run edge MTP segment_a
  isend hidden to cloud
  suspend MTPDraftState
  mtp_draft_inflight_count += 1

MTP_DRAFT_LAST:
  irecv cloud hidden
  validate head_token
  resume MTPDraftState
  run edge MTP segment_e
  generate draft token id
  mtp_draft_inflight_count -= 1
```

## 8.5 多 draft step

当 `num_speculative_tokens > 1` 时，按 step 串行：

```text
step0 MTP_DRAFT_LAST 产出 draft_token_0
  -> 创建 step1 MTP_DRAFTTask
step1 MTP_DRAFT_LAST 产出 draft_token_1
  -> 创建 step2 MTP_DRAFTTask
step2 MTP_DRAFT_LAST 产出 draft_token_2
  -> 汇总 draft token ids
  -> Scheduler.update_draft_token_ids
```

---

# 9. 队列与 in-flight 计数

## 9.1 边侧新增队列

| 队列 | 含义 |
|---|---|
| `mtp_draft_waiting[]` | 已创建、尚未下发的 Qwen-MTP draft task |
| `mtp_drafts_last_ready[]` | 云侧返回后等待执行 edge last 的 draft task |
| `mtp_draft_result_ready[]` | 已生成 draft token ids、等待回填 Scheduler |

## 9.2 云侧新增队列

| 队列 | 来源 | 含义 |
|---|---|---|
| `ready_mtp_drafts[]` | `MTP_DRAFT_FIRST` | 可执行 MTP_DRAFT中 |

## 9.3 `mtp_draft_inflight_count`

语义：

```text
已经发出 MTP_DRAFT_FIRST，但还没完成 MTP_DRAFT_LAST 的 MTP_DRAFT batch 数量。
```

它是全局边侧调度器计数，不是单请求计数。

计数变化：

| 调度动作 | 计数变化 |
|---|---|
| 调度 `MTP_DRAFT_FIRST` | `mtp_draft_inflight_count += 1` |
| 完成 `MTP_DRAFT_LAST` | `mtp_draft_inflight_count -= 1` |
| abort 清理未完成 draft | 按 task 状态释放 |

## 9.4 与 `num_speculative_tokens` 的区别

```text
num_speculative_tokens = 3
表示单请求每轮需要串行生成 3 个 draft token。

mtp_draft_inflight_limit = 1/2/3
表示全局最多允许几个 MTP_DRAFT batch 同时跨边云在飞。
```

首版建议：

```python
mtp_draft_inflight_limit = 1
```

原因：

1. 同一请求内部 draft step 串行依赖。
2. 多请求 draft 并发会引入乱序回包和 state 管理复杂度。
3. 首版优先保证 Qwen-MTP 功能正确性。

---

# 10. 边侧调度状态机

## 10.1 D首可调度条件

表里的 `D首 / VERIFY_DECODE_FIRST` 只表示“已经具备 verify 条件”的请求。

```python
def can_schedule_verify_decode_first(req):
    if not qwen_mtp_enabled(req):
        return True
    return req.spec_token_ids_ready()
```

| 请求类型 | 是否可作为 D首 调度 |
|---|---|
| 非 MTP 请求 | 保持原逻辑 |
| Qwen-MTP 请求，draft tokens 已 ready | 可以调度 |
| Qwen-MTP 请求，存在 pending / inflight MTP_DRAFT | 不可以调度 |
| Qwen-MTP 请求，刚完成 P_LAST 但还没生成首轮 draft | 不可以调度，应先创建并执行 MTP_DRAFT |

## 10.2 IDLE

| 优先级 | Batch | 来源 | 状态机变化 |
|---|---|---|---|
| 1 | P首 / chunk0首 | `waiting[]` | `IDLE -> LOW` |
| 2 | MTP_DRAFT首 | `mtp_draft_waiting[]` | `IDLE` |
| 3 | MTP_DRAFT尾 | `mtp_drafts_last_ready[]` | `IDLE` |
| 4 | D首，即可调度的 `VERIFY_DECODE_FIRST` | `running[]` 中 draft ready 的请求 | `IDLE` |
| 5 | D尾，即 `VERIFY_DECODE_LAST` | `decodes_last_ready[]` | `IDLE` |
| 6 | Empty | - | `IDLE` |

## 10.3 LOW

| 优先级 | Batch | 来源 | 状态机变化 |
|---|---|---|---|
| 1 | chunk(i>0)首 | `chunk_prefill_first[]` | `LOW -> HIGH` |
| 2 | P首 | `waiting[]` | `LOW -> HIGH` |
| 3 | P尾 / chunk(i)尾 | `prefills_last_ready[]` | `LOW -> IDLE` |
| 4 | MTP_DRAFT首 | `mtp_draft_waiting[]` | `LOW` |
| 5 | MTP_DRAFT尾 | `mtp_drafts_last_ready[]` | `LOW` |
| 6 | D首，即可调度的 `VERIFY_DECODE_FIRST` | `running[]` 中 draft ready 的请求 | `LOW` |
| 7 | D尾，即 `VERIFY_DECODE_LAST` | `decodes_last_ready[]` | `LOW` |
| 8 | Empty | - | `LOW` |

## 10.4 HIGH

| 优先级 | Batch | 来源 | 状态机变化 |
|---|---|---|---|
| 1 | P尾 / chunk(i)尾 | `prefills_last_ready[]` | `HIGH -> LOW` |
| 2 | MTP_DRAFT首 | `mtp_draft_waiting[]` | `HIGH` |
| 3 | MTP_DRAFT尾 | `mtp_drafts_last_ready[]` | `HIGH` |
| 4 | D首，即可调度的 `VERIFY_DECODE_FIRST` | `running[]` 中 draft ready 的请求 | `HIGH` |
| 5 | D尾，即 `VERIFY_DECODE_LAST` | `decodes_last_ready[]` | `HIGH` |
| 6 | Empty | - | `HIGH` |

---

# 11. 云侧调度状态机

云侧仍保留原 P/D 期望状态机：

```text
EXPECT_EXECUTE_PREFILL
EXPECT_EXECUTE_DECODE
```

新增 `ready_mtp_drafts[]`。

## 11.1 EXPECT_EXECUTE_PREFILL

| 优先级 | Batch | 来源 | 状态变化 |
|---|---|---|---|
| 1 | P中 | `ready_prefills[]` | `EEP -> EED` |
| 2 | MTP_DRAFT中 | `ready_mtp_drafts[]` | `EEP` |
| 3 | D中，即 `VERIFY_DECODE_MIDDLE` | `ready_decodes[]` | `EEP` |
| 4 | Empty | - | `EEP` |

## 11.2 EXPECT_EXECUTE_DECODE

| 优先级 | Batch | 来源 | 状态变化 |
|---|---|---|---|
| 1 | MTP_DRAFT中 | `ready_mtp_drafts[]` | `EED` |
| 2 | D中，即 `VERIFY_DECODE_MIDDLE` | `ready_decodes[]` | `EED -> EEP` |
| 3 | P中 | `ready_prefills[]` | `EED` |
| 4 | Empty | - | `EED` |

说明：

1. `EXPECT_EXECUTE_PREFILL` 下，P中 仍最高，命中后状态转为 `EXPECT_EXECUTE_DECODE`。
2. 除 P中 命中场景外，MTP_DRAFT中 优先于 D中。
3. `EXPECT_EXECUTE_DECODE` 下，MTP_DRAFT中 优先于 D中，用于优先补齐 Qwen-MTP 下一轮 VERIFY 所需的 draft tokens。
4. 只有真正执行 D中 时，状态才从 `EED -> EEP`。

## 11.3 MTP_DRAFT 中段是否切片

首版建议：

```text
MTP_DRAFT中 不做 layerwise slicing。
```

原因：

1. Qwen-MTP draft middle 本身计算较短。
2. P中 slicing 已经依赖 `_layerwise_intermediate` 等状态。
3. MTP_DRAFT 中段如果切片，必须新增独立 continuation state，不能复用 P中 的 layerwise state。

---

# 12. Worker / ModelRunner 执行路径

## 12.1 边侧派发

| batch type | 边侧动作 |
|---|---|
| `PREFILL_FIRST` | segment_a，发送 P hidden，挂起 HeadState |
| `PREFILL_LAST` | 接收 P hidden，恢复 HeadState，segment_e + sampler |
| `DECODE_FIRST` | segment_a，发送 verify hidden，挂起 HeadState |
| `DECODE_LAST` | 接收 verify hidden，恢复 HeadState，segment_e + sampler / accept-reject |
| `MTP_DRAFT_FIRST` | Qwen-MTP edge first，发送 draft hidden，挂起 MTPDraftState |
| `MTP_DRAFT_LAST` | 接收 draft hidden，恢复 MTPDraftState，生成 draft token ids |

## 12.2 云侧派发

| batch type | 云侧动作 |
|---|---|
| `PREFILL_FIRST` | 接收 P hidden，执行 P middle，发送 P hidden 回边 |
| `DECODE_FIRST` | 接收 verify hidden，执行 verify middle，发送 hidden 回边 |
| `MTP_DRAFT_FIRST` | 接收 draft hidden，执行 Qwen-MTP middle，发送 draft hidden 回边 |

## 12.3 `sample_tokens()` 职责收缩

迁移后：

```text
sample_tokens()
  -> 主模型采样 / accept-reject / 请求状态输出
  -> 创建 MTPDraftTask
  -> 不同步执行完整 Qwen-MTP proposer
```

`MTP_DRAFT_LAST` 不走普通用户 sampler；它只生成 draft token ids。

## 12.4 draft token ids 回填

```text
MTP_DRAFT_LAST
  -> 写入 draft token ids buffer
  -> EngineCore / Scheduler take_draft_token_ids
  -> Scheduler.update_draft_token_ids
  -> 下一轮 DECODE_FIRST 填充 scheduled_spec_decode_tokens
```

---

# 13. EngineCore 修改点

## 13.1 POST_OUT 消费

边侧处理 POST_OUT 时需要分类：

```python
if so.batch_type == BatchType.PREFILL_LAST:
    scheduler.prefills_last_ready.append(so)
elif so.batch_type == BatchType.DECODE_LAST:
    scheduler.decodes_last_ready.append(so)
elif so.batch_type == BatchType.MTP_DRAFT_LAST:
    scheduler.mtp_drafts_last_ready.append(so)
```

## 13.2 PRE_OUT publish

只允许 first 类 batch 发往云侧：

```text
PREFILL_FIRST
DECODE_FIRST
MTP_DRAFT_FIRST
```

禁止发送：

```text
PREFILL_LAST
DECODE_LAST
MTP_DRAFT_LAST
EMPTY
```

## 13.3 `_needs_sample_tokens`

边云模式下：

```python
def _needs_sample_tokens(scheduler_output):
    return scheduler_output.batch_type in (
        BatchType.PREFILL_LAST,
        BatchType.DECODE_LAST,
    )
```

`MTP_DRAFT_LAST` 不进入普通 `sample_tokens()`，它走 draft result path。

## 13.4 触发 MTPDraftTask

`P_LAST` 和 `VERIFY_DECODE_LAST` 都可能触发 MTPDraftTask：

```text
P_LAST:
  产出首个真实 token
  若请求继续生成，创建首轮 MTP_DRAFT

VERIFY_DECODE_LAST:
  产出用户 token / accept-reject
  若请求继续生成，创建下一轮 MTP_DRAFT
```

---

# 14. 通信风险与处理策略

| 风险 | 说明 | 处理 |
|---|---|---|
| 控制面和数据面乱序 | POST_OUT 先到，但 PP hidden 对不上 | `head_token == _head_token` 强校验 |
| MTP_DRAFT_LAST 假 ready | 控制面到达但 hidden 未到，边侧阻塞等待 | 推荐 MTP_DRAFT_LAST 在 cloud middle 完成并发起 isend 后 POST_OUT |
| metadata RTT 放大 | `num_speculative_tokens=3` 会产生多次小包往返 | 采集 metadata 阻塞时间，后续考虑 schema 缓存 |
| channel 串包 | 复用 DECODE channel 容易和 verify 混淆 | 新增 `HiddenChannelType.MTP_DRAFT` |
| 多请求 draft 并发乱序 | 多个 MTP task 同时在飞，last 回来顺序不固定 | head_token + draft_task_id + draft_step_idx 三重校验 |
| abort 后回包到达 | request 已取消，draft hidden 或控制面仍返回 | 回填前校验 request，清理 stale state |
| send handle 等待阻塞 | 等待 draft send handle 影响 P/D 尾段处理 | MTP_DRAFT 独立 send handle，避免等待所有 channel |

---

# 15. 分阶段实施计划

## 15.1 Phase M1：本地语义拆分

目标：

- `sample_tokens()` 不同步执行完整 Qwen-MTP proposer。
- `VERIFY_DECODE_LAST` 只产出用户 token 和请求状态。
- 未结束请求创建 pending draft task。
- draft 结果仍可通过本地 bridge 回填 Scheduler。

## 15.2 Phase M2：BatchType / SchedulerOutput / 队列接入

目标：

- 新增 `MTP_DRAFT_FIRST/LAST`。
- 新增 `ready_mtp_drafts[]`、`mtp_draft_waiting[]`、`mtp_drafts_last_ready[]`。
- 新增 `MTPDraftTask` 和 `MTPDraftState`。
- 明确 `head_token` / `_head_token` 配对协议。

## 15.3 Phase M3：MTP_DRAFT 首中尾跨边云链路

目标：

- Worker 支持 Qwen-MTP edge first / cloud middle / edge last。
- MTP_DRAFT 走 PRE_OUT / POST_OUT 控制面。
- MTP_DRAFT hidden states 走 PP/HCCL 数据面。
- MTP_DRAFT_LAST 写 draft token ids buffer。

## 15.4 Phase M4：边侧/云侧调度策略接入

目标：

- 边侧 IDLE/LOW/HIGH 状态机加入 MTP_DRAFT。
- 云侧 EEP/EED ready queue 加入 `ready_mtp_drafts[]`。
- `DECODE_FIRST` 对 Qwen-MTP 请求增加 draft ready 过滤。

## 15.5 Phase M5：通信优化与性能调优

目标：

- 评估 `mtp_draft_inflight_limit > 1`。
- 评估 MTP_DRAFT POST_OUT 发送时机。
- 采集 metadata RTT、MTP_DRAFT 往返耗时和 draft miss。
- 评估多 draft step 合并或 graph capture。

---

# 16. 验证方案

## 16.1 功能验证

| 场景 | 期望 |
|---|---|
| `speculative_config=None` | 完全不进入 MTP_DRAFT，输出和原路径一致 |
| Qwen-MTP + `num_speculative_tokens=1` | draft/verify 闭环可运行 |
| Qwen-MTP + `num_speculative_tokens=3` | 三步 draft 后 verify width=4 |
| abort during draft | draft result 不回填 |
| 多请求混合 | 某请求 draft 不阻塞其他请求 P_LAST/D_LAST |

## 16.2 通信验证

需要日志或 trace 能看到：

```text
MTP_DRAFT_FIRST PRE_OUT publish
MTP_DRAFT_FIRST edge isend hidden with _head_token
MTP_DRAFT_MIDDLE cloud irecv / middle / isend
MTP_DRAFT_LAST POST_OUT publish
MTP_DRAFT_LAST edge irecv hidden
head_token == _head_token
draft_token_ids 回填 Scheduler
```

需要采集：

```text
MTP_DRAFT edge->cloud hidden size
MTP_DRAFT cloud->edge hidden size
metadata blocking time
MTP_DRAFT_LAST 等待 hidden time
mtp_draft_inflight_count
ready_mtp_drafts queue length
mtp_drafts_last_ready queue length
```

## 16.3 调度验证

需要日志或 trace 能看到：

```text
P_FIRST/P_MIDDLE/P_LAST
MTP_DRAFT_FIRST/MIDDLE/LAST step=0
MTP_DRAFT_FIRST/MIDDLE/LAST step=1
MTP_DRAFT_FIRST/MIDDLE/LAST step=2
DECODE_FIRST/DECODE_MIDDLE/DECODE_LAST  # semantic VERIFY_DECODE
```

---

# 17. 正确性不变量汇总

```text
1. P_LAST 可以触发首轮 MTP_DRAFT。
2. VERIFY_DECODE_LAST 可以触发后续轮 MTP_DRAFT。
3. MTP_DRAFT_LAST 不产生用户可见 token。
4. 用户可见 token 只由 P_LAST 或 VERIFY_DECODE_LAST 产生。
5. MTP_DRAFT 不改变请求 finished 状态。
6. MTP_DRAFT result 回填前必须校验 request 仍存在且未 abort。
7. MTP_DRAFT 使用独立 head_token。
8. MTP_DRAFT 数据面必须携带 _head_token。
9. SchedulerOutput.head_token 必须等于 IntermediateTensors["_head_token"]。
10. DECODE_FIRST/DECODE_LAST 对非 MTP 请求语义不变。
11. speculative_config is None 时不进入任何 MTP_DRAFT 逻辑。
12. speculative_config.method != "mtp" 时不进入 Qwen-MTP 路径。
13. MTP_DRAFT_FIRST 才允许进入 PRE_OUT。
14. MTP_DRAFT_LAST 不允许进入 PRE_OUT。
15. 云侧不处理 MTP_DRAFT_LAST。
16. MTP_DRAFT中 不复用 P中 layerwise continuation state。
```

---

# 18. 风险点汇总

| 风险 | 说明 | 处理 |
|---|---|---|
| draft 阻塞边侧尾段处理 | 若仍在 `sample_tokens()` 内同步跑完整 Qwen-MTP draft，会卡住其他尾段任务 | 拆成 `MTP_DRAFT` 派生任务 |
| 首轮 draft 触发点遗漏 | 只从 `VERIFY_LAST` 触发会漏掉 `P_LAST` 后的首轮 draft | `P_LAST` 和 `VERIFY_LAST` 都需要能创建 draft task |
| channel 错配 | MTP draft 复用 DECODE channel 容易和主 verify 串包 | 新增 `MTP_DRAFT` channel |
| draft miss | draft 未及时 ready，下一轮 verify 只能 width=1 | 首版允许统计，后续优化优先级和 in-flight |
| 多 step 通信放大 | `num_speculative_tokens=3` 会有 3 次 draft 往返 | 串行 step 先保证正确性，再评估合并 |
| Graph/metadata stale | Qwen-MTP cloud attention metadata 依赖 positions/spec_step_idx | 复用来源刷新逻辑，step 维度显式传递 |
| abort stale result | 请求取消后 draft 才返回 | 回填前校验 request id / generation |
| MTP_DRAFT 控制面早到 | 边侧优先执行 MTP_DRAFT_LAST 但 hidden 未到 | 推荐 data-ready 后 POST_OUT 或记录等待时间 |

