# PDbatch 分离边云协同 Qwen-MTP 迁移调度设计文档

本文档聚焦 `vllm-ascend` 中 Qwen-MTP 的已有实现，以及该实现如何接入 PDMIX 边云 PD 掩盖调度框架。

参考：

- [PDbatch分离分布式边云协同推理调度算法设计.md](PDbatch分离分布式边云协同推理调度算法设计.md)
- [PDbatch分离边云协同Phase5&7详细设计.md](PDbatch分离边云协同Phase5&7详细设计.md)
- [PDbatch分离边云协同Phase2-4 extend详细设计.md](PDbatch分离边云协同Phase2-4 extend详细设计.md)

---

# 1. 设计范围

本文档关注：

1. PDMIX 调度框架支持 Qwen-MTP 开启后的 decode verify + draft 生成闭环。
2. `VERIFY_DECODE_LAST` 不再同步阻塞等待 Qwen-MTP draft 生成。
3. Qwen-MTP draft 作为独立派生任务进入边云调度框架。
4. draft 结果通过 `take_draft_token_ids()` 或等价路径回填 Scheduler。
5. 下一轮 `VERIFY_DECODE` 消费 `scheduled_spec_decode_tokens`。
6. 在 PD 掩盖场景下，保留原有 P 首尾优先级和 P/D 掩盖节奏。

核心问题不是新增一套独立 decode 主流程，而是把 Qwen-MTP 的 draft/verify 生命周期接进现有 PDMIX PD 掩盖调度。

---

# 2. 背景与核心认知

当前 PDMIX 已把一次主模型执行拆为：

```text
P_FIRST(edge) -> P_MIDDLE(cloud) -> P_LAST(edge)
D_FIRST(edge) -> D_MIDDLE(cloud) -> D_LAST(edge)
```

原 PD 掩盖要隐藏的 decode 气泡是：

```text
D_MIDDLE(k)
    -> 云到边传输
    -> 边侧 D_LAST: 尾段 + lm_head + sample
    -> 边侧 D_FIRST: 下一 token head
    -> 边到云传输
    -> D_MIDDLE(k+1)
```

通过在两个 `D_MIDDLE` 之间插入 `P_MIDDLE slice`，云侧可以减少等待边侧首尾和传输的空转。

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

---

# 3. Qwen-MTP 来源实现摘要

来源仓为 `vllm-ascend`，本文只参考 Qwen-MTP 相关路径。

## 3.1 关键文件

| 文件 | 作用 |
|---|---|
| `vllm_ascend/spec_decode/llm_base_proposer.py` | Qwen-MTP proposer 主逻辑，包含 `_run_mtp_edge_cloud()`、多 draft step 循环、draft token 计算 |
| `vllm_ascend/patch/worker/patch_qwen3_next_mtp.py` | Qwen3 Next MTP 相关 patch，主要处理 KV cache 绑定兼容 |
| `vllm_ascend/patch/worker/patch_qwen3_dflash.py` | DFlash Qwen3 相关 patch，处理 context KV 预计算 |
| `vllm_ascend/worker/model_runner_v1.py` | 当前 `sample_tokens()` 同步触发 `propose_draft_token_ids()` 的入口 |

## 3.2 Qwen-MTP 边云三段

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

来源代码中 `_run_mtp_edge_cloud()` 当前是同步函数：edge 执行 first 后立即等待 cloud middle，再继续 edge last。这个组织方式在单体或非 PD 掩盖路径可以成立，但迁移到 PDMIX 后会阻塞边侧尾段处理。

---

# 4. 为什么要拆 draft 和 verify

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
2. 同一个边侧 worker 上其他请求的 `P_LAST` / `VERIFY_LAST` 尾段处理也会被推迟。

这不是“同一个请求下一轮 verify 依赖 draft”本身的问题。这个依赖一定存在。真正要避免的是：

```text
某个请求的 draft 生成占住 VERIFY_LAST 调用栈，
导致边侧无法及时处理其他已经 ready 的尾段。
```

迁移后的目标是：

```text
VERIFY_LAST 只负责产出用户 token 和请求状态；
若请求未结束，则创建 MTP_DRAFT task；
MTP_DRAFT task 后续由调度器按优先级执行。
```

---

# 5. 一次请求流水

以下以 `num_speculative_tokens = 3` 为例。

## 5.1 Prefill 后首轮 draft

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

随后继续：

```text
MTP_DRAFT(round=2, step=0)
MTP_DRAFT(round=2, step=1)
MTP_DRAFT(round=2, step=2)
VERIFY_DECODE(round=2, width=1+3)
```

稳定形态是：

```text
DRAFT_1 -> DRAFT_2 -> DRAFT_3 -> VERIFY(width=1+3)
```

这里的 `DRAFT_1/2/3` 是下一轮 `VERIFY` 的前置准备，不是和下一轮 `VERIFY` 无依赖的并行 filler。

---

# 6. 需要掩盖的气泡

不带 MTP 时，主要掩盖：

```text
D_MIDDLE(k)
    -> D_LAST(k)
    -> D_FIRST(k+1)
    -> D_MIDDLE(k+1)
```

Qwen-MTP 加入后，要掩盖的气泡变成更细的几段：

| 气泡 | 说明 |
|---|---|
| `P_LAST -> MTP_DRAFT(step0)_MIDDLE ready` | Prefill 结束后首轮 draft 启动前的边侧处理和传输 |
| `MTP_DRAFT(step0)_MIDDLE -> step1_MIDDLE` | draft step 之间的边侧 last/first 与通信 |
| `MTP_DRAFT(step1)_MIDDLE -> step2_MIDDLE` | draft step 之间的边侧 last/first 与通信 |
| `MTP_DRAFT(step2)_MIDDLE -> VERIFY_MIDDLE ready` | draft 完成、回填 Scheduler、verify first 发送到云侧 |
| `VERIFY_MIDDLE -> next MTP_DRAFT(step0)_MIDDLE` | verify 尾段完成后触发下一轮 draft |

时间长短判断：

```text
从单轮 VERIFY 到下一轮 VERIFY 看，链路更长，因为多了 draft step。
从平均每个输出 token 看，如果接受率高，MTP 可以更短。
```

Qwen-MTP draft 的云侧计算通常短于主模型 verify middle，但每个 step 仍有边云往返、调度和同步成本。因此调度重点不是“draft 算得重”，而是避免 draft 的通信同步把边侧尾段处理卡住。

---

# 7. Batch Type 与 HiddenChannel

## 7.1 Batch Type

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

原因：

1. 目标仓已有 `DECODE_FIRST / DECODE_LAST` 边云分离链路。
2. Qwen-MTP 的主模型 verify 仍是 decode 请求。
3. 不额外改 vLLM Scheduler 的主 decode batch type 语义。
4. draft 输出不是用户 token，必须新增独立 batch type，不能复用 `DECODE_LAST`。

## 7.2 SchedulerOutput 字段

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

## 7.3 HiddenChannel

需要新增：

```python
HiddenChannelType.MTP_DRAFT
```

不建议复用 `HiddenChannelType.DECODE`，原因：

1. `DECODE` channel 已服务主模型 verify。
2. Qwen-MTP draft 有独立首尾生命周期。
3. 复用容易导致 `head_token` 数据面错配。
4. 复用会让 draft send/recv handle 反向阻塞 verify 尾段处理。

---

# 8. 队列与状态

## 8.1 边侧新增队列

| 队列 | 含义 |
|---|---|
| `mtp_draft_waiting[]` | 已创建、尚未下发的 Qwen-MTP draft task |
| `mtp_drafts_last_ready[]` | 云侧返回后等待执行 edge last 的 draft task |
| `mtp_draft_result_ready[]` | 已生成 draft token ids、等待回填 Scheduler |

保留原队列：

| 队列 | 含义 |
|---|---|
| `waiting[]` | 等待 prefill 的请求 |
| `running[]` | 已完成 prefill，可做主模型 verify decode 的请求 |
| `prefills_last_ready[]` | P_LAST ready |
| `decodes_last_ready[]` | VERIFY_DECODE_LAST ready |

## 8.2 in-flight 计数

新增：

```python
mtp_draft_inflight_count
mtp_draft_inflight_limit = 1
```

首版保持保守：

```text
同一时刻最多 1 个 MTP_DRAFT 跨边云在飞。
```

不变量：

```text
prefill_inflight_count 只由 P_FIRST/P_LAST 修改。
decode_inflight_count 只由 DECODE_FIRST/DECODE_LAST 修改。
mtp_draft_inflight_count 只由 MTP_DRAFT_FIRST/MTP_DRAFT_LAST 修改。
```

---

# 9. 边侧调度优先级

采用保守原则：

```text
不改变原 PDMIX 的 P首/P尾、D首/D尾相对优先级；
但 D首 的可调度条件需要区分普通 decode 和 Qwen-MTP verify。
```

关键修正：

```text
表里的 D首 / VERIFY_DECODE_FIRST 只表示“已经具备 verify 条件”的请求。
如果 Qwen-MTP 请求还在等待 draft token ids，则不能作为 D首 被选中。
```

否则会出现：

```text
P_LAST 产出首 token
    -> 请求进入 running[]
    -> 边侧优先调度 D首
    -> 但 draft 还没生成
    -> 本轮 VERIFY 只能 width=1
```

这会绕过 Qwen-MTP 的核心收益。正确做法是：

```text
Qwen-MTP 请求需要先完成 MTP_DRAFT，拿到 draft token ids；
然后这个请求才成为可调度的 VERIFY_DECODE_FIRST。
```

因此，第 9 章的优先级表有两个判断层：

1. 先按原 PDMIX 状态机选择 batch 类型优先级。
2. 对 `D首 / VERIFY_DECODE_FIRST` 做 eligibility 过滤：只有 draft ready 的 Qwen-MTP 请求才可被选中。

## 9.1 D首 可调度条件

```python
def can_schedule_verify_decode_first(req):
    if not qwen_mtp_enabled(req):
        return True

    return req.spec_token_ids_ready()
```

更完整地说：

| 请求类型 | 是否可作为 D首 调度 |
|---|---|
| 非 MTP 请求 | 保持原逻辑 |
| Qwen-MTP 请求，draft tokens 已 ready | 可以调度 `VERIFY_DECODE_FIRST` |
| Qwen-MTP 请求，存在 pending / inflight MTP_DRAFT | 不可以调度 `VERIFY_DECODE_FIRST` |
| Qwen-MTP 请求，刚完成 P_LAST 但还没生成首轮 draft | 不可以调度 `VERIFY_DECODE_FIRST`，应先创建并执行 `MTP_DRAFT` |

这不是把普通 decode 和 MTP decode 做两套流程，而是保证：

```text
MTP 模式下的 decode 请求进入 verify 前，必须先满足 draft 前置条件。
```

## 9.2 MTP_DRAFT 的插入位置

`MTP_DRAFT` 不直接覆盖原有 P/D 优先级。

在 IDLE 状态下，采用：

```text
P首 > MTP_DRAFT首 > MTP_DRAFT尾 > VERIFY_DECODE首 > VERIFY_DECODE尾
```

原因：

```text
IDLE 状态下边侧没有 prefill 在飞，优先启动 P首 保持云侧后续 P中 供给；
没有 P首 可启动时，优先启动 Qwen-MTP draft 链，避免后续 VERIFY 频繁缺少 draft token ids。
```

MTP 类内部先写首段：

```text
MTP_DRAFT首 > MTP_DRAFT尾
```

这里不依赖 MTP_DRAFT 的前身是谁，而是从 IDLE 状态的调度目标出发：

```text
尽快启动下一轮 draft 链，使后续 VERIFY_DECODE 有 draft tokens 可消费。
```

## 9.3 IDLE

原始优先级：

```text
P首 / chunk0首 > D首 > D尾 > Empty
```

加入 Qwen-MTP 后：

| 优先级 | Batch | 来源 | 状态机变化 |
|---|---|---|---|
| 1 | P首 / chunk0首 | `waiting[]` | `IDLE -> LOW` |
| 2 | MTP_DRAFT首 | `mtp_draft_waiting[]` | `IDLE` |
| 3 | MTP_DRAFT尾 | `mtp_drafts_last_ready[]` | `IDLE` |
| 4 | D首，即可调度的 `VERIFY_DECODE_FIRST` | `running[]` 中 draft ready 的请求 | `IDLE` |
| 5 | D尾，即 `VERIFY_DECODE_LAST` | `decodes_last_ready[]` | `IDLE` |
| 6 | Empty | - | `IDLE` |

说明：

```text
IDLE 下 MTP 类优先于 Verify 类。
如果 running[] 中只有等待 draft 的 Qwen-MTP 请求，它不能作为 D首 被选中。
```

## 9.4 LOW

原始优先级：

```text
chunk(i>0)首 > P首 > P尾 > D首 > D尾 > Empty
```

加入 Qwen-MTP 后：

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

说明：

```text
P首、P尾、chunk 首尾仍按原 PDMIX LOW 状态优先级执行。
非 P 类任务中，MTP 类优先于 Verify 类。
MTP 类内部按 MTP_DRAFT首 > MTP_DRAFT尾 排序。
```

## 9.5 HIGH

原始优先级：

```text
P尾 > D首 > D尾 > Empty
```

加入 Qwen-MTP 后：

| 优先级 | Batch | 来源 | 状态机变化 |
|---|---|---|---|
| 1 | P尾 / chunk(i)尾 | `prefills_last_ready[]` | `HIGH -> LOW` |
| 2 | MTP_DRAFT首 | `mtp_draft_waiting[]` | `HIGH` |
| 3 | MTP_DRAFT尾 | `mtp_drafts_last_ready[]` | `HIGH` |
| 4 | D首，即可调度的 `VERIFY_DECODE_FIRST` | `running[]` 中 draft ready 的请求 | `HIGH` |
| 5 | D尾，即 `VERIFY_DECODE_LAST` | `decodes_last_ready[]` | `HIGH` |
| 6 | Empty | - | `HIGH` |

说明：

```text
HIGH 状态下不能继续调度 P首，因此优先处理 P尾 释放 prefill 在飞数。
P尾 之后，MTP 类优先于 Verify 类，避免 Qwen-MTP 请求长期缺少 draft token ids。
MTP 类内部按 MTP_DRAFT首 > MTP_DRAFT尾 排序。
```

核心差异不在表面顺序，而在 `D首` 的可调度条件：

```text
普通请求：D首 仍按原逻辑调度。
Qwen-MTP 请求：只有 draft token ids ready 后，D首 才能调度。
```

---

# 10. 云侧调度优先级

云侧仍保留原 P/D 期望状态机：

```text
EXPECT_EXECUTE_PREFILL
EXPECT_EXECUTE_DECODE
```

新增 `ready_mtp_drafts[]`，先沿用 EEP/EED 两态完成 Qwen-MTP draft 的云侧接入。

## 10.1 EXPECT_EXECUTE_PREFILL

| 优先级 | Batch | 来源 | 状态变化 |
|---|---|---|---|
| 1 | P中 | `ready_prefills[]` | `EEP -> EED` |
| 2 | MTP_DRAFT中 | `ready_mtp_drafts[]` | `EEP` |
| 3 | D中，即 `VERIFY_DECODE_MIDDLE` | `ready_decodes[]` | `EEP` |
| 4 | Empty | - | `EEP` |

## 10.2 EXPECT_EXECUTE_DECODE

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

---

# 11. 调度与调用链修改点

## 11.1 Scheduler

需要新增：

```text
mtp_draft_waiting[]
mtp_drafts_last_ready[]
mtp_draft_inflight_count
```

需要扩展：

```text
schedule() 能构造 MTP_DRAFT_FIRST。
处理 channel inbox 时能把 MTP_DRAFT_LAST 放入 mtp_drafts_last_ready[]。
MTP_DRAFT_LAST 完成后能回填 draft token ids。
```

不能改：

```text
普通 prefill/decode 请求生命周期。
非 MTP 请求的 running/waiting 状态迁移。
```

## 11.2 EngineCore

需要新增：

```text
处理 VERIFY_DECODE_LAST 输出之后，触发或接收 MTP_DRAFT task。
处理 MTP_DRAFT_LAST 输出之后，调用 Scheduler.update_draft_token_ids。
```

关键要求：

```text
VERIFY_DECODE_LAST 的 ModelRunnerOutput 必须先进入 update_from_output；
draft 生成不能阻塞这个 update。
```

## 11.3 Worker

边侧需要支持：

| batch type | 动作 |
|---|---|
| `MTP_DRAFT_FIRST` | 执行 Qwen-MTP edge first，发送 hidden 到云侧，挂起 MTPDraftState |
| `MTP_DRAFT_LAST` | 接收云侧 hidden，恢复 MTPDraftState，执行 Qwen-MTP edge last，写 draft token ids |

云侧需要支持：

| batch type | 动作 |
|---|---|
| `MTP_DRAFT_FIRST` | 接收 edge hidden，执行 Qwen-MTP middle，发送 hidden 回边侧 |

## 11.4 ModelRunner

需要拆分：

```text
sample_tokens()
    只做主模型采样 / accept-reject / 请求状态输出
    不同步调用 Qwen-MTP proposer 完整链路
```

新增：

```text
build_qwen_mtp_draft_task()
run_qwen_mtp_draft_first()
run_qwen_mtp_draft_last()
take_draft_token_ids()
```

复用来源：

```text
Qwen-MTP model input 构造
Qwen-MTP attention metadata 刷新
draft token ids 生成逻辑
spec_step_idx 多步循环语义
```

调度框架侧避免复用：

```text
不复用普通 ExecuteModelState 表示 MTPDraftState。
不复用普通 DECODE channel 表示 MTP_DRAFT 数据面。
不复用 prefill layerwise continuation state 表示 MTP_DRAFT 中段。
```

---

# 12. `num_speculative_tokens` 策略

Qwen-MTP 的 `num_speculative_tokens` 来自 `speculative_config.num_speculative_tokens`，不是写死值。

支持：

```text
num_speculative_tokens >= 1
```

但实现策略保持串行 step：

```text
for draft_step_idx in range(num_speculative_tokens):
    MTP_DRAFT_FIRST(step=draft_step_idx)
    MTP_DRAFT_MIDDLE(step=draft_step_idx)
    MTP_DRAFT_LAST(step=draft_step_idx)
```

原因：

1. Qwen-MTP 后一个 draft token 依赖前一个 draft token。
2. 串行 step 更接近来源实现中的多 draft step 循环。
3. 每个 step 独立进入调度框架，便于 PD 掩盖插入 P中。

建议测试主配置：

```text
num_speculative_tokens = 3
```

同时保留 `1` 的基础回归。

---

# 13. 正确性不变量

```text
1. P_LAST 可以触发首轮 MTP_DRAFT。
2. VERIFY_DECODE_LAST 可以触发后续轮 MTP_DRAFT。
3. MTP_DRAFT_LAST 不产生用户可见 token。
4. 用户可见 token 只由 P_LAST 或 VERIFY_DECODE_LAST 产生。
5. MTP_DRAFT 不改变请求 finished 状态。
6. MTP_DRAFT result 回填前必须校验 request 仍存在且未 abort。
7. MTP_DRAFT 使用独立 head_token。
8. MTP_DRAFT 使用独立 HiddenChannelType.MTP_DRAFT。
9. DECODE_FIRST/DECODE_LAST 对非 MTP 请求语义不变。
10. speculative_config is None 时不进入任何 MTP_DRAFT 逻辑。
11. speculative_config.method != "mtp" 时不进入 Qwen-MTP 路径。
```

---

# 14. 风险点

| 风险 | 说明 | 处理 |
|---|---|---|
| draft 阻塞边侧尾段处理 | 若仍在 `sample_tokens()` 内同步跑完整 Qwen-MTP draft，会卡住其他尾段任务 | 拆成 `MTP_DRAFT` 派生任务 |
| 首轮 draft 触发点遗漏 | 只从 `VERIFY_LAST` 触发会漏掉 `P_LAST` 后的首轮 draft | `P_LAST` 和 `VERIFY_LAST` 都需要能创建 draft task |
| channel 错配 | MTP draft 复用 DECODE channel 容易和主 verify 串包 | 新增 `MTP_DRAFT` channel |
| draft miss | draft 未及时 ready，下一轮 verify 只能 width=1 | 首版允许，但需统计 miss；后续优化优先级 |
| P/D 原调度被破坏 | MTP_DRAFT 过早抢占 P首/P尾，会影响已有 PD 掩盖收益 | 采用保守优先级 |
| 多 step 通信放大 | `num_speculative_tokens=3` 会有 3 次 draft 往返 | 串行 step 先保证正确性，再评估合并 |
| Graph/metadata stale | Qwen-MTP cloud attention metadata 依赖 positions/spec_step_idx | 复用来源刷新逻辑，step 维度显式传递 |
| abort stale result | 请求取消后 draft 才返回 | 回填前校验 request id / generation |

---

# 15. 分阶段实施

## 15.1 Phase M1：本地语义拆分

目标：

- `sample_tokens()` 不同步执行完整 Qwen-MTP proposer。
- `VERIFY_DECODE_LAST` 只产出用户 token 和请求状态。
- 未结束请求创建 pending draft task。
- draft 结果仍可通过本地 bridge 回填 Scheduler。

验收：

```text
非 MTP 请求行为不变。
Qwen-MTP 请求能产出用户 token。
draft token ids 能回填。
VERIFY_DECODE_LAST 调用栈不包含完整 draft 生成。
```

## 15.2 Phase M2：Qwen-MTP_DRAFT 首尾分离

目标：

- 新增 `MTP_DRAFT_FIRST/LAST`。
- 新增 `HiddenChannelType.MTP_DRAFT`。
- Worker 支持 Qwen-MTP edge first / cloud middle / edge last。
- Scheduler 支持 `mtp_draft_waiting[]` 和 `mtp_drafts_last_ready[]`。

验收：

```text
Qwen-MTP draft 可跨边云执行。
P_LAST 后能生成首轮 draft。
VERIFY_LAST 后能生成后续轮 draft。
下一轮 VERIFY 可消费 scheduled_spec_decode_tokens。
```

## 15.3 Phase M3：保守云侧接入

目标：

- PassiveScheduler 新增 `ready_mtp_drafts[]`。
- 保留 EEP/EED 两态。
- fallback 支持 `MTP_DRAFT中`。

验收：

```text
P中/D中 原有期望状态机不被破坏。
MTP_DRAFT中 可以在云侧被调度。
无 MTP_DRAFT 尾段等待长期堆积。
```

## 15.4 Phase M4：性能调优

目标：

- 统计 draft miss。
- 统计尾段等待堆积。
- 根据 profile 决定是否提升 MTP_DRAFT 优先级。
- 评估多 draft step 合并或 graph capture。

---

# 16. 验证建议

## 16.1 功能验证

| 场景 | 期望 |
|---|---|
| `speculative_config=None` | 完全不进入 MTP_DRAFT，输出和原路径一致 |
| `method="mtp"` + Qwen + `num_speculative_tokens=1` | draft/verify 闭环可运行 |
| `method="mtp"` + Qwen + `num_speculative_tokens=3` | 三步 draft 后 verify width=4 |
| abort during draft | draft result 不回填 |
| 多请求混合 | 某请求 draft 不阻塞其他请求 P_LAST/D_LAST |

## 16.2 调度验证

需要日志或 trace 能看到：

```text
P_FIRST/P_MIDDLE/P_LAST
MTP_DRAFT_FIRST/MIDDLE/LAST step=0
MTP_DRAFT_FIRST/MIDDLE/LAST step=1
MTP_DRAFT_FIRST/MIDDLE/LAST step=2
DECODE_FIRST/DECODE_MIDDLE/DECODE_LAST  # semantic VERIFY_DECODE
```

需要确认：

```text
P/D 原状态机优先级不变。
MTP_DRAFT 使用独立 head_token。
MTP_DRAFT 使用独立 channel。
draft token ids 在下一轮 decode 前回填。
```

---

# 17. 总结

核心是完成 Qwen-MTP 在 PDMIX 中的调度接入：

```text
主请求 decode = VERIFY_DECODE
派生 draft = MTP_DRAFT
P_LAST / VERIFY_LAST 触发 draft
MTP_DRAFT 跨边云生成 draft token ids
Scheduler 回填 draft token ids
下一轮 VERIFY_DECODE 消费 scheduled_spec_decode_tokens
```

调度策略采用保守版本：

```text
保留原 P/D 首尾优先级；
MTP_DRAFT 独立队列、独立 inflight、独立 channel；
云侧先以 EEP/EED 两态接入 ready_mtp_drafts；
性能优先级提升根据 trace 决定。
```
