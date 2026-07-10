# PDbatch 分离边云协同 MTP 迁移调度设计文档

参考：

- [PDbatch分离分布式边云协同推理调度算法设计.md](PDbatch分离分布式边云协同推理调度算法设计.md)
- [PDbatch分离边云协同Phase5&7详细设计.md](PDbatch分离边云协同Phase5&7详细设计.md)
- [PDbatch分离边云协同Phase2-4 extend详细设计.md](PDbatch分离边云协同Phase2-4 extend详细设计.md)

---

# 1. 背景

当前 PD 分离边云协同框架已经把一次主模型执行拆成：

```text
P首(edge) -> P中(cloud) -> P尾(edge)
D首(edge) -> D中(cloud) -> D尾(edge)
```

并通过：

- `PREFILL_FIRST / PREFILL_LAST`
- `DECODE_FIRST / DECODE_LAST`
- `prefills_last_ready[] / decodes_last_ready[]`
- `head_token`
- `HiddenChannelType.PREFILL_1 / PREFILL_2 / DECODE`
- `prefill_inflight_count / decode_inflight_count`

实现 P/D 首尾分离和 2P1D 掩盖调度。

MTP 开启后，decode 阶段不应被理解为“普通 decode”和“MTP decode”两套流程切换。
在 vllm-ascend speculative decoding 设计中，Scheduler 仍然调度 decode 请求，只是该 decode 请求会携带上一轮 proposer 生成的 `scheduled_spec_decode_tokens`。
目标模型在本轮 decode 中对这些 draft tokens 进行 verify，边侧在尾段完成 logits、采样、accept/reject，并产生用户可见 token。

因此 PD 分离场景下需要把 decode 拆成两类子任务：

| 子任务 | 含义 | 输出 |
|---|---|---|
| `VERIFY_DECODE` | 主模型验证阶段，消费 `scheduled_spec_decode_tokens`，执行目标模型 decode verify | 用户可见 token、请求状态、是否继续生成 |
| `MTP_DRAFT` | MTP proposer 阶段，由 `VERIFY_DECODE_LAST` 触发，生成下一轮 draft token ids | draft token ids，不产生用户 token |

原 vllm-ascend 中 `propose_draft_token_ids()` 在 `sample_tokens()` 内同步执行。
这个假设在 PD 分离下不成立：MTP proposer 本身也可能包含边侧和云侧分段计算，如果继续作为 `DECODE_LAST.sample_tokens()` 的同步本地函数调用，会阻塞边侧 tail drain，破坏 PD 掩盖流水线。

---

# 2. 设计目标

## 2.1 功能目标

1. MTP 开启后，主模型 decode 始终按 `VERIFY_DECODE` 语义执行。
2. `VERIFY_DECODE` 仍复用原 Scheduler 的 decode 调度和 `scheduled_spec_decode_tokens` 数据契约。
3. `VERIFY_DECODE_LAST` 只完成主模型 verify、lm_head、sampler、accept/reject、请求状态更新。
4. MTP proposer 从 `sample_tokens()` 内拆出，成为异步 `MTP_DRAFT` 任务。
5. `MTP_DRAFT_LAST` 完成后，通过 `take_draft_token_ids()` 或等价机制把 draft token ids 回填 Scheduler，供下一轮 `VERIFY_DECODE` 使用。
6. PD 掩盖优先保证 tail drain，避免 MTP draft 生成阻塞 P尾/D尾。

## 2.2 性能目标

目标流水从：

```text
VERIFY_D中 -> VERIFY_D尾 + MTP_DRAFT同步执行 + VERIFY_D首 -> VERIFY_D中
```

调整为：

```text
VERIFY(k)_中 -> VERIFY(k)_尾
    -> MTP_DRAFT(k+1, step=0)_首 -> MTP_DRAFT(k+1, step=0)_中 -> MTP_DRAFT(k+1, step=0)_尾
    -> MTP_DRAFT(k+1, step=1)_首 -> MTP_DRAFT(k+1, step=1)_中 -> MTP_DRAFT(k+1, step=1)_尾
    -> ...
    -> MTP_DRAFT(k+1, step=N-1)_首 -> MTP_DRAFT(k+1, step=N-1)_中 -> MTP_DRAFT(k+1, step=N-1)_尾
    -> VERIFY(k+1, width=1+N)_首 -> VERIFY(k+1, width=1+N)_中
```

即：

```text
P中 用于掩盖 VERIFY 尾段、DRAFT step 间、DRAFT 到下一轮 VERIFY 之间的边侧计算与传输气泡。
MTP_DRAFT 不是普通 filler，而是下一轮 VERIFY 的前置任务。
```

## 2.3 非目标

- 不把 MTP 设计成普通 decode 与 MTP decode 两套主流程切换。
- 不让 `MTP_DRAFT` 产生用户可见 token。
- 不复用 `DECODE_LAST` self-posting 优化处理 MTP draft。
- 不在本阶段改写 vLLM 原生 speculative decoding 对 `scheduled_spec_decode_tokens` 的基本语义。

---

# 3. 术语

| 术语 | 含义 |
|---|---|
| `VERIFY_DECODE_FIRST` | 主模型 verify decode 首段，边侧 segment_a |
| `VERIFY_DECODE_MIDDLE` | 主模型 verify decode 中段，云侧 segment_b/c |
| `VERIFY_DECODE_LAST` | 主模型 verify decode 尾段，边侧 segment_e + lm_head + sampler + accept/reject |
| `MTP_DRAFT_FIRST` | MTP proposer 首段，边侧 MTP embedding/fc 或头部计算 |
| `MTP_DRAFT_MIDDLE` | MTP proposer 中段，云侧 MTP decoder layer |
| `MTP_DRAFT_LAST` | MTP proposer 尾段，边侧 MTP norm/lm_head/argmax，输出 draft token ids |
| `verify_head_token` | VERIFY_DECODE 首尾配对 token |
| `draft_head_token` | MTP_DRAFT 首尾配对 token |
| `parent_head_token` | MTP_DRAFT 关联的 VERIFY_DECODE_LAST token，用于追踪从哪次 verify 派生 |

实现上可以新增独立 batch type，也可以在已有 `SchedulerOutput` 上增加 subtype 字段。
为了避免和普通 decode 语义混淆，推荐显式新增 batch type。

---

# 4. Batch Type 设计

## 4.1 推荐新增 batch type

```python
BatchType.VERIFY_DECODE_FIRST
BatchType.VERIFY_DECODE_LAST
BatchType.MTP_DRAFT_FIRST
BatchType.MTP_DRAFT_LAST
```

兼容关系：

| 旧 batch type | MTP 开启后语义 |
|---|---|
| `DECODE_FIRST` | 可作为 `VERIFY_DECODE_FIRST` 的兼容别名 |
| `DECODE_LAST` | 可作为 `VERIFY_DECODE_LAST` 的兼容别名 |

推荐第一阶段保守实现：

```text
DECODE_FIRST / DECODE_LAST 仍保留名称，但内部语义明确标记为 VERIFY_DECODE。
MTP_DRAFT_FIRST / MTP_DRAFT_LAST 必须新增，不能复用 DECODE_FIRST / DECODE_LAST。
```

原因：

- 主模型 verify 与 MTP draft 都是 decode shape，但输出语义不同；
- `VERIFY_DECODE_LAST` 输出用户 token；
- `MTP_DRAFT_LAST` 输出 draft token ids；
- 两者不能共用 `update_from_output()` 的请求状态更新逻辑。

## 4.2 SchedulerOutput 新增字段

建议新增：

```python
is_mtp_draft: bool = False
draft_step_idx: int = 0
parent_head_token: str | None = None
draft_task_id: str | None = None
```

字段语义：

| 字段 | 含义 |
|---|---|
| `is_mtp_draft` | 当前 batch 是否为 MTP proposer 任务 |
| `draft_step_idx` | 多 token MTP 中的 draft step 序号 |
| `parent_head_token` | 触发该 MTP_DRAFT 的 VERIFY_DECODE_LAST token |
| `draft_task_id` | MTP_DRAFT 自身全局唯一 id，可等价使用 `head_token` |

`MTP_DRAFT_FIRST` 必须携带：

```text
head_token       # 本 draft 首尾配对
parent_head_token
draft_step_idx
hidden_channel
```

---

# 5. 数据结构与队列设计

## 5.1 边侧队列

在 `PDSeparatedScheduler` 基础上新增：

| 队列 | 含义 | 是否参与主 decode 调度 |
|---|---|---|
| `mtp_draft_waiting[]` | `VERIFY_DECODE_LAST` 触发但尚未下发的 MTP_DRAFT 任务 | 否 |
| `mtp_drafts_last_ready[]` | 云侧返回的 MTP_DRAFT_LAST 控制面 | 否 |
| `mtp_draft_result_ready[]` | MTP_DRAFT_LAST 已生成的 draft token ids，待回填 Scheduler | 否 |

原有队列保持：

| 队列 | 含义 |
|---|---|
| `running[]` | 已完成 P尾、可进行 VERIFY_DECODE 的请求 |
| `decodes_last_ready[]` | VERIFY_DECODE_LAST ready |
| `prefills_last_ready[]` | PREFILL_LAST ready |

## 5.2 in-flight 计数

新增：

```python
mtp_draft_inflight_count
mtp_draft_inflight_limit
```

推荐默认：

```python
mtp_draft_inflight_limit = 1
```

计数变化：

| 调度动作 | 计数变化 |
|---|---|
| 调度 `MTP_DRAFT_FIRST` | `mtp_draft_inflight_count += 1` |
| 完成 `MTP_DRAFT_LAST` | `mtp_draft_inflight_count -= 1` |

不变量：

```text
0 <= mtp_draft_inflight_count <= mtp_draft_inflight_limit
```

## 5.3 HiddenChannel 设计

不建议 MTP_DRAFT 复用普通 `DECODE` channel。

推荐新增：

```python
HiddenChannelType.MTP_DRAFT
```

原因：

1. `VERIFY_DECODE` 与 `MTP_DRAFT` 都是 decode shape，但生命周期不同。
2. `VERIFY_DECODE_LAST` 需要尽快 drain，不能被 MTP_DRAFT 的 send/recv handle 阻塞。
3. `DECODE` channel 当前存在 self-posting 优化假设，MTP_DRAFT 不满足该假设。
4. 独立 channel 可以清晰隔离 `head_token` 与 PP 数据面配对关系。

如果硬件或通信实现暂时不能新增 channel，则必须满足：

```text
decode_inflight_count + mtp_draft_inflight_count <= 1
```

这会牺牲 MTP_DRAFT 与 VERIFY_DECODE 的重叠能力，只适合作为过渡方案。

---

# 6. 边侧状态机设计

## 6.1 状态变量

沿用 PrefillState：

| 状态 | 条件 |
|---|---|
| `IDLE` | `prefill_inflight_count == 0` |
| `LOW` | `prefill_inflight_count == 1` |
| `HIGH` | `prefill_inflight_count >= prefill_inflight_limit` |

新增 MTP_DRAFT 维度：

```text
mtp_draft_inflight_count
len(mtp_draft_waiting)
len(mtp_drafts_last_ready)
```

MTP_DRAFT 不改变 PrefillState。
PrefillState 仍只由 prefill in-flight 数决定。

## 6.2 边侧调度优先级总原则

优先级原则：

```text
保留原 PD 边侧 P 首尾状态机优先级；
MTP 只插入 decode 类任务内部。
```

具体含义：

1. `PREFILL_FIRST` / `PREFILL_LAST` 的相对优先级沿用 Phase5/7，不因 MTP 引入而整体前移或后移。
2. `IDLE` 时仍优先启动新的 `PREFILL_FIRST`，保持 P中 供给。
3. `LOW` 时仍优先处理 `PREFILL_LAST`、chunk 后续首段、第二个 `PREFILL_FIRST`。
4. `HIGH` 时仍优先处理 `PREFILL_LAST`，禁止继续调度 `PREFILL_FIRST`。
5. 在 decode 类内部，`VERIFY_DECODE_LAST` 高于 `MTP_DRAFT_LAST`，因为它产生用户可见 token 并解除主 decode in-flight。
6. `MTP_DRAFT_LAST` 高于新的 first 段，因为它使下一轮 verify 能携带 draft tokens。
7. first 段中默认 `MTP_DRAFT_FIRST` 高于 `VERIFY_DECODE_FIRST`，避免下一轮 verify 等 draft 或退化为 width=1。
8. `VERIFY_DECODE_FIRST` 受 `decode_inflight_count < decode_inflight_limit` 限制；`MTP_DRAFT_FIRST` 受 `mtp_draft_inflight_count < mtp_draft_inflight_limit` 限制。

## 6.3 IDLE 状态优先级

条件：

```text
prefill_inflight_count == 0
```

| 优先级 | Batch Type | 请求来源 | 状态变化 |
|---|---|---|---|
| 1 | P首 / chunk0首 | `waiting[]` / `chunk_prefill_first[]` | `IDLE -> LOW` |
| 2 | VERIFY_D尾 | `decodes_last_ready[]` | `IDLE -> IDLE` |
| 3 | MTP_DRAFT尾 | `mtp_drafts_last_ready[]` | `IDLE -> IDLE` |
| 4 | MTP_DRAFT首 | `mtp_draft_waiting[]` | `IDLE -> IDLE` |
| 5 | VERIFY_D首 | `running[]` | `IDLE -> IDLE` |
| 6 | Empty | - | `IDLE -> IDLE` |

说明：

- IDLE 继承原 Phase5/7 策略，优先启动新的 Prefill；
- decode 类内部按 `VERIFY_D尾 -> MTP_DRAFT尾 -> MTP_DRAFT首 -> VERIFY_D首` 排序；
- `MTP_DRAFT首` 高于 `VERIFY_D首` 是为了优先补齐下一轮 verify 的 draft 前置条件。

## 6.4 LOW 状态优先级

条件：

```text
prefill_inflight_count == 1
```

| 优先级 | Batch Type | 请求来源 | 状态变化 |
|---|---|---|---|
| 1 | P尾 / chunk(i)尾 | `prefills_last_ready[]` | `LOW -> IDLE` |
| 2 | chunk(i>0)首 | `chunk_prefill_first[]` | `LOW -> HIGH` |
| 3 | P首 | `waiting[]` | `LOW -> HIGH` |
| 4 | VERIFY_D尾 | `decodes_last_ready[]` | `LOW -> LOW` |
| 5 | MTP_DRAFT尾 | `mtp_drafts_last_ready[]` | `LOW -> LOW` |
| 6 | MTP_DRAFT首 | `mtp_draft_waiting[]` | `LOW -> LOW` |
| 7 | VERIFY_D首 | `running[]` | `LOW -> LOW` |
| 8 | Empty | - | `LOW -> LOW` |

说明：

- LOW 继承原 Phase5/7 策略：先释放 P 在飞槽位，再补 chunk 后续首段或第二个 P首；
- MTP 不抢占 `chunk(i>0)首` 和 `P首`，避免破坏原 2P1D 掩盖节奏；
- decode 类内部仍按 `VERIFY_D尾 -> MTP_DRAFT尾 -> MTP_DRAFT首 -> VERIFY_D首` 排序。

## 6.5 HIGH 状态优先级

条件：

```text
prefill_inflight_count >= prefill_inflight_limit
```

| 优先级 | Batch Type | 请求来源 | 状态变化 |
|---|---|---|---|
| 1 | P尾 / chunk(i)尾 | `prefills_last_ready[]` | `HIGH -> LOW` |
| 2 | VERIFY_D尾 | `decodes_last_ready[]` | `HIGH -> HIGH` |
| 3 | MTP_DRAFT尾 | `mtp_drafts_last_ready[]` | `HIGH -> HIGH` |
| 4 | MTP_DRAFT首 | `mtp_draft_waiting[]` | `HIGH -> HIGH` |
| 5 | VERIFY_D首 | `running[]` | `HIGH -> HIGH` |
| 6 | Empty | - | `HIGH -> HIGH` |

说明：

- HIGH 禁止继续调度 P首；
- 此时云侧 P中供给已经足够，边侧可以发起 MTP_DRAFT首补充下一轮 draft；
- VERIFY_D首 仍受 `decode_inflight_count < 1` 限制；
- MTP_DRAFT首 受 `mtp_draft_inflight_count < mtp_draft_inflight_limit` 限制。

## 6.6 边侧调度伪代码

```python
def schedule_edge():
    drain_post_out()

    state = get_prefill_state()

    if state == IDLE:
        if can_schedule_prefill_first():
            return pick_prefill_first()
        if decodes_last_ready:
            return pick_verify_decode_last()
        if mtp_drafts_last_ready:
            return pick_mtp_draft_last()
        if can_schedule_mtp_draft_first():
            return pick_mtp_draft_first()
        if can_schedule_verify_decode_first():
            return pick_verify_decode_first()
        return empty()

    if state == LOW:
        if prefills_last_ready:
            return pick_prefill_last()
        if can_schedule_prefill_first():
            return pick_prefill_first()
        if decodes_last_ready:
            return pick_verify_decode_last()
        if mtp_drafts_last_ready:
            return pick_mtp_draft_last()
        if can_schedule_mtp_draft_first():
            return pick_mtp_draft_first()
        if can_schedule_verify_decode_first():
            return pick_verify_decode_first()
        return empty()

    if state == HIGH:
        if prefills_last_ready:
            return pick_prefill_last()
        if decodes_last_ready:
            return pick_verify_decode_last()
        if mtp_drafts_last_ready:
            return pick_mtp_draft_last()
        if can_schedule_mtp_draft_first():
            return pick_mtp_draft_first()
        if can_schedule_verify_decode_first():
            return pick_verify_decode_first()
        return empty()
```

其中：

```python
def can_schedule_verify_decode_first():
    return bool(running) and decode_inflight_count < decode_inflight_limit

def can_schedule_mtp_draft_first():
    return (
        bool(mtp_draft_waiting)
        and mtp_draft_inflight_count < mtp_draft_inflight_limit
        and mtp_draft_channel_available()
    )
```

---

# 7. VERIFY_DECODE 调度规则

## 7.1 VERIFY_DECODE_FIRST

来源：

```text
running[]
```

输入：

```text
request.output_token_ids
request.spec_token_ids
scheduled_spec_decode_tokens
```

调度动作：

1. 从 `running[]` 构造主模型 decode SchedulerOutput；
2. 若请求有上一轮 draft tokens，则填充 `scheduled_spec_decode_tokens`；
3. 标记：

```python
scheduler_output.batch_type = BatchType.VERIFY_DECODE_FIRST
scheduler_output.head_token = uuid4().hex
scheduler_output.hidden_channel = HiddenChannelType.DECODE
decode_inflight_count += 1
```

兼容阶段可继续使用：

```python
BatchType.DECODE_FIRST
```

但必须在代码注释和状态机中明确其语义是 `VERIFY_DECODE_FIRST`。

## 7.2 VERIFY_DECODE_LAST

来源：

```text
decodes_last_ready[]
```

执行动作：

1. edge recv 云侧 hidden；
2. resume VERIFY HeadState；
3. 执行 segment_e；
4. 计算 logits；
5. sampler 完成 accept/reject；
6. 更新请求状态；
7. 输出用户可见 token；
8. 若请求未结束，创建 `MTP_DRAFT` 任务。

计数变化：

```python
decode_inflight_count -= 1
```

创建 MTP_DRAFT 条件：

```text
speculative_config.method == "mtp"
请求未 finished
请求仍需要继续生成 token
本轮采样结果可作为 proposer 输入
```

注意：

```text
VERIFY_DECODE_LAST 不同步执行 MTP proposer。
```

它只把 MTP_DRAFT task 放入 `mtp_draft_waiting[]`，然后尽快返回 `ModelRunnerOutput`。

---

# 8. MTP_DRAFT 调度规则

## 8.1 MTP_DRAFT task 数据结构

建议新增：

```python
@dataclass
class MTPDraftTask:
    task_id: str
    parent_head_token: str
    req_ids: list[str]
    sampled_token_ids: torch.Tensor | list[list[int]]
    sampling_metadata: Any
    spec_decode_metadata: Any
    spec_decode_common_attn_metadata: Any
    positions: torch.Tensor
    target_hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor | None
    batch_desc: Any
    num_scheduled_tokens: int
```

字段来源：

| 字段 | 来源 |
|---|---|
| `sampled_token_ids` | VERIFY_DECODE_LAST sampler 输出 |
| `target_hidden_states` | VERIFY_DECODE_LAST 主模型 hidden states，必要时使用 `get_mtp_target_hidden_states()` |
| `spec_decode_common_attn_metadata` | VERIFY_DECODE_LAST 构造的 common attn metadata |
| `positions` | VERIFY_DECODE_LAST positions |
| `batch_desc` | VERIFY_DECODE_LAST batch descriptor |

## 8.2 MTP_DRAFT_FIRST

来源：

```text
mtp_draft_waiting[]
```

调度动作：

```python
task = mtp_draft_waiting.popleft()
scheduler_output = make_mtp_draft_scheduler_output(task)
scheduler_output.batch_type = BatchType.MTP_DRAFT_FIRST
scheduler_output.head_token = task.task_id
scheduler_output.parent_head_token = task.parent_head_token
scheduler_output.hidden_channel = HiddenChannelType.MTP_DRAFT
mtp_draft_inflight_count += 1
```

边侧执行：

```text
MTP embed/fc/head segment -> send to cloud -> suspend MTPDraftState -> return EMPTY
```

## 8.3 MTP_DRAFT_MIDDLE

云侧执行：

```text
recv MTP_DRAFT_FIRST tensors
build MTP draft attn metadata
run MTP decoder layer for draft_step_idx
send hidden back to edge
publish MTP_DRAFT_LAST
```

云侧必须保留并回传：

```text
head_token
parent_head_token
draft_step_idx
```

数据面 `IntermediateTensors` 必须携带：

```text
_head_token
_parent_head_token
_draft_step_idx
```

## 8.4 MTP_DRAFT_LAST

来源：

```text
mtp_drafts_last_ready[]
```

边侧执行：

1. recv 云侧 MTP hidden；
2. resume MTPDraftState；
3. 执行 MTP tail segment；
4. 计算 draft logits；
5. greedy/argmax 得到 draft token ids；
6. 写入 `DraftTokenIds` 缓冲；
7. `mtp_draft_inflight_count -= 1`。

输出：

```text
draft token ids
```

不输出：

```text
用户 token
logprobs
request finished status
```

## 8.5 多 speculative token

当 `num_speculative_tokens > 1` 时，有两种实现策略。

### 策略 A：一个 MTP_DRAFT task 内部串行多步

```text
MTP_DRAFT_FIRST(step0) -> MTP_DRAFT_LAST(step0)
MTP_DRAFT_FIRST(step1) -> MTP_DRAFT_LAST(step1)
...
```

优点：

- 与来源项目 `_run_merged_draft` 逻辑接近；
- 每个 step 可独立携带 `draft_step_idx`。

缺点：

- task 管理复杂；
- 多 step draft 可能延长 draft ready 时间。

### 策略 B：一个 MTP_DRAFT batch 内携带多 step

```text
MTP_DRAFT_FIRST(all steps) -> MTP_DRAFT_MIDDLE(loop) -> MTP_DRAFT_LAST(all steps)
```

优点：

- 控制面 batch 数少；
- 更接近原 `propose_draft_token_ids()` 一次返回 `[B, num_speculative_tokens]` 的接口。

缺点：

- 云侧 middle 会被一个 MTP task 占用更久；
- 对 PD 掩盖更不友好。

推荐第一阶段采用 **策略 A**，优先保证调度可控和 tail drain。

---

# 9. 云侧调度状态机修改

## 9.1 ready queue

在 PassiveScheduler 中新增：

| 队列 | 来源 | 含义 |
|---|---|---|
| `ready_prefills[]` | `PREFILL_FIRST` | P中 |
| `ready_decodes[]` | `VERIFY_DECODE_FIRST` | VERIFY_D中 |
| `ready_mtp_drafts[]` | `MTP_DRAFT_FIRST` | MTP_DRAFT中 |
| `ready_pdmixes[]` | legacy `PD_MIX` | 兼容旧路径 |

接收规则：

| batch type | 云侧行为 |
|---|---|
| `PREFILL_FIRST` | 放入 `ready_prefills[]` |
| `VERIFY_DECODE_FIRST` / `DECODE_FIRST` | 放入 `ready_decodes[]` |
| `MTP_DRAFT_FIRST` | 放入 `ready_mtp_drafts[]` |
| `PREFILL_LAST` | 丢弃并记录错误 |
| `VERIFY_DECODE_LAST` / `DECODE_LAST` | 丢弃并记录错误 |
| `MTP_DRAFT_LAST` | 丢弃并记录错误 |
| `EMPTY` | 丢弃 |

## 9.2 云侧状态

原状态：

```python
EXPECT_EXECUTE_PREFILL
EXPECT_EXECUTE_DECODE
```

加入 MTP 后推荐扩展为：

```python
EXPECT_EXECUTE_PREFILL
EXPECT_EXECUTE_VERIFY
EXPECT_EXECUTE_MTP_DRAFT
```

状态目标：

| 状态 | 优先目标 |
|---|---|
| `EXPECT_EXECUTE_PREFILL` | 调度 P中，维持 PD 掩盖材料 |
| `EXPECT_EXECUTE_VERIFY` | 调度 VERIFY_D中，推动用户 token |
| `EXPECT_EXECUTE_MTP_DRAFT` | 调度 MTP_DRAFT中，准备下一轮 draft |

## 9.3 云侧优先级

云侧不能把 `VERIFY_D中 > MTP_DRAFT中 > P中` 简化成无状态全局优先级。
原因是原 PD 掩盖依赖期望状态机保持 P中 供给：当状态明确期望 `P中`
且 `P中` 已 ready 时，应允许它命中状态并推进状态机。

加入 MTP 后的原则是：

```text
命中当前期望类型时优先执行期望类型；
fallback 时按 VERIFY_D中 > MTP_DRAFT中 > P中；
fallback 不改变状态，下一轮继续尝试当前期望类型。
```

这样可以同时保证：

1. `VERIFY_D中` 不被 draft 或 prefill 推迟，优先产生用户可见 token；
2. `MTP_DRAFT中` 不被 P slice 长期压住，避免下一轮 verify 等 draft 或退化为 width=1；
3. `P中` 在 `EXPECT_EXECUTE_PREFILL` 命中时仍可补充 PD 掩盖材料。

### EXPECT_EXECUTE_PREFILL

| 优先级 | Batch | 来源 | 状态变化 |
|---|---|---|---|
| 1 | P中 | `ready_prefills[]` | `EEP -> EEV` |
| 2 | VERIFY_D中 | `ready_decodes[]` | `EEP -> EEP` |
| 3 | MTP_DRAFT中 | `ready_mtp_drafts[]` | `EEP -> EEP` |
| 4 | Empty | - | `EEP -> EEP` |

说明：

- 当前状态期望 P中，若 P中 ready，则优先执行 P中 并进入期望 VERIFY；
- 若 P中 不 ready，fallback 按 VERIFY_D中、MTP_DRAFT中 的顺序执行；
- fallback 后保持 EEP，下一轮继续尝试 P中，避免 P 被 decode 类任务长期饿死。

### EXPECT_EXECUTE_VERIFY

| 优先级 | Batch | 来源 | 状态变化 |
|---|---|---|---|
| 1 | VERIFY_D中 | `ready_decodes[]` | `EEV -> EEM` |
| 2 | MTP_DRAFT中 | `ready_mtp_drafts[]` | `EEV -> EEV` |
| 3 | P中 | `ready_prefills[]` | `EEV -> EEV` |
| 4 | Empty | - | `EEV -> EEV` |

说明：

- 当前状态期望 VERIFY，若 VERIFY_D中 ready，则优先执行并进入期望 MTP_DRAFT；
- 若 VERIFY_D中 不 ready，MTP_DRAFT中 高于 P中；
- 这里不让 P中 抢在 MTP_DRAFT中 前面，因为 draft 是下一轮 VERIFY 的前置任务，不是普通填空任务。

### EXPECT_EXECUTE_MTP_DRAFT

| 优先级 | Batch | 来源 | 状态变化 |
|---|---|---|---|
| 1 | MTP_DRAFT中 | `ready_mtp_drafts[]` | `EEM -> EEP` |
| 2 | VERIFY_D中 | `ready_decodes[]` | `EEM -> EEM` |
| 3 | P中 | `ready_prefills[]` | `EEM -> EEM` |
| 4 | Empty | - | `EEM -> EEM` |

说明：

- 当前状态期望 MTP_DRAFT，若 MTP_DRAFT中 ready，则优先执行并回到期望 P中；
- 若 MTP_DRAFT中 不 ready，VERIFY_D中 高于 P中；
- fallback 后保持 EEM，下一轮继续尝试 MTP_DRAFT，避免 draft 长期被 P slice 或 VERIFY fallback 饿死。

说明：

- 状态只在命中期望类型时转移；
- fallback 调度时状态保持不变，避免某类 batch 长期饿死；
- 在非期望命中的 fallback 路径上，`VERIFY_D中` 优先于 `MTP_DRAFT中`，`MTP_DRAFT中` 优先于 `P中`；
- `MTP_DRAFT中` 不应长期饿死，否则下一轮 VERIFY_DECODE 会等待 draft 或退化为无 draft 验证。

## 9.4 云侧调度伪代码

```python
def schedule_cloud():
    poll_and_classify()

    if state == EXPECT_EXECUTE_PREFILL:
        if ready_prefills:
            state = EXPECT_EXECUTE_VERIFY
            return pop_prefill()
        if ready_decodes:
            return pop_verify_decode()
        if ready_mtp_drafts:
            return pop_mtp_draft()
        return empty()

    if state == EXPECT_EXECUTE_VERIFY:
        if ready_decodes:
            state = EXPECT_EXECUTE_MTP_DRAFT
            return pop_verify_decode()
        if ready_mtp_drafts:
            return pop_mtp_draft()
        if ready_prefills:
            return pop_prefill()
        return empty()

    if state == EXPECT_EXECUTE_MTP_DRAFT:
        if ready_mtp_drafts:
            state = EXPECT_EXECUTE_PREFILL
            return pop_mtp_draft()
        if ready_decodes:
            return pop_verify_decode()
        if ready_prefills:
            return pop_prefill()
        return empty()
```

## 9.5 云侧切片策略

| batch | 是否切片 | 说明 |
|---|---|---|
| `PREFILL_FIRST` | 是 | 继续作为主要 PD 掩盖材料 |
| `VERIFY_DECODE_FIRST` | 否 | decode shape，保持短任务 |
| `MTP_DRAFT_FIRST` | 默认否 | 第一阶段避免与 prefill layerwise continuation 状态冲突 |

MTP_DRAFT 是否切片后续可扩展，但必须新增独立 continuation state：

```python
_layerwise_mtp_draft_intermediate
_layerwise_mtp_draft_scheduler_output
_layerwise_mtp_draft_positions
_layerwise_mtp_draft_attn_metadata
```

不能复用 prefill 的：

```python
_layerwise_intermediate
```

否则 P中 slice 与 MTP_DRAFT中 slice 交错时会互相覆盖。

---

# 10. EngineCore 修改点

## 10.1 `_needs_sample_tokens`

当前逻辑：

```python
return bt in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST)
```

加入 MTP 后应调整：

```python
def _needs_sample_tokens(scheduler_output):
    bt = scheduler_output.batch_type
    return bt in (
        BatchType.PREFILL_LAST,
        BatchType.VERIFY_DECODE_LAST,
        BatchType.DECODE_LAST,          # compat alias
    )
```

`MTP_DRAFT_LAST` 不走普通 `sample_tokens()`，它走 draft result path：

```python
if bt == BatchType.MTP_DRAFT_LAST:
    return False
```

原因：

- MTP_DRAFT_LAST 不执行用户 sampler；
- 它只生成 draft token ids。

## 10.2 `_maybe_publish_pre_out`

需要发送到云侧的 first 段：

```python
if bt in (
    BatchType.PREFILL_FIRST,
    BatchType.VERIFY_DECODE_FIRST,
    BatchType.DECODE_FIRST,        # compat alias
    BatchType.MTP_DRAFT_FIRST,
):
    publish(scheduler_output)
```

不能发送：

```text
PREFILL_LAST
VERIFY_DECODE_LAST
MTP_DRAFT_LAST
EMPTY
```

## 10.3 `_drain_pd_channel_inbox`

需要分类：

```python
if bt == BatchType.PREFILL_LAST:
    scheduler.prefills_last_ready.append(so)
elif bt in (BatchType.VERIFY_DECODE_LAST, BatchType.DECODE_LAST):
    scheduler.decodes_last_ready.append(so)
elif bt == BatchType.MTP_DRAFT_LAST:
    scheduler.mtp_drafts_last_ready.append(so)
```

## 10.4 `take_draft_token_ids`

原路径中 `take_draft_token_ids()` 从 `sample_tokens()` 内同步生成的 `_draft_token_ids` 读取。

迁移后：

```text
MTP_DRAFT_LAST -> 写 DraftTokenIds 缓冲 -> EngineCore.take_draft_token_ids() -> Scheduler.update_draft_token_ids_in_output()
```

需要保证：

1. `MTP_DRAFT_LAST` 完成后 draft token ids 已经可取；
2. Scheduler 在下一轮 `VERIFY_DECODE_FIRST` 前能消费该结果；
3. 若 draft 还没 ready，则下一轮 verify 可以无 draft 运行，但这属于 draft miss，不是模式回退。

---

# 11. Worker 修改点

## 11.1 batch_type 派发表

边侧：

| batch type | worker 动作 |
|---|---|
| `PREFILL_FIRST` | edge head segment，send P hidden |
| `PREFILL_LAST` | recv P hidden，edge tail segment，sample |
| `VERIFY_DECODE_FIRST` / `DECODE_FIRST` | edge head segment，send verify hidden |
| `VERIFY_DECODE_LAST` / `DECODE_LAST` | recv verify hidden，edge tail segment，sample/accept/reject |
| `MTP_DRAFT_FIRST` | edge MTP draft head，send draft hidden |
| `MTP_DRAFT_LAST` | recv draft hidden，edge MTP draft tail，write draft token ids |

云侧：

| batch type | worker 动作 |
|---|---|
| `PREFILL_FIRST` | recv P hidden，run P middle，send P hidden back |
| `VERIFY_DECODE_FIRST` / `DECODE_FIRST` | recv verify hidden，run verify middle，send hidden back |
| `MTP_DRAFT_FIRST` | recv MTP draft hidden，run MTP middle，send draft hidden back |

## 11.2 send handle 隔离

现有实现按 hidden channel 维护：

```python
_pp_send_work_by_channel
```

加入 MTP 后必须保证：

```text
PREFILL_1/PREFILL_2/DECODE/MTP_DRAFT 分别 wait 自己的 channel。
```

不允许在 `MTP_DRAFT_FIRST` 入口等待所有 channel 的 send work。
否则 MTP_DRAFT 会反向阻塞 VERIFY_D尾/P尾 drain。

---

# 12. ModelRunner 修改点

## 12.1 VERIFY_DECODE_LAST 拆分

原 `sample_tokens()` 中：

```python
sampler_output = self._sample(...)
...
propose_draft_token_ids(...)
...
return ModelRunnerOutput(...)
```

迁移后：

```python
sampler_output = self._sample(...)
...
if mtp_enabled and request_not_finished:
    task = build_mtp_draft_task(...)
    self.enqueue_mtp_draft_task(task)
...
return ModelRunnerOutput(...)
```

即：

```text
sample_tokens() 不再同步调用 MTP proposer。
```

## 12.2 MTPDraftState

新增类似 `HeadState` 的挂起态：

```python
@dataclass
class MTPDraftState:
    head_token: str
    parent_head_token: str
    task: MTPDraftTask
    positions: torch.Tensor
    attn_metadata: Any
    batch_desc: Any
    target_hidden_states: torch.Tensor
    next_token_ids: torch.Tensor
```

用途：

```text
MTP_DRAFT_FIRST 完成边侧 first segment 后挂起；
MTP_DRAFT_LAST recv 云侧 hidden 后恢复，用于生成 draft token ids。
```

## 12.3 edge-cloud MTP proposer 复用范围

可复用来源项目逻辑：

- `AscendEagleProposer` 中 MTP 分支；
- `_run_merged_draft` 中的 MTP hidden states / logits 生成；
- `attn_update_stack_num_spec_norm` 对多 draft step 的 seq_lens/slot_mapping 更新；
- `compute_draft_token_ids()`；
- DeepSeek/Qwen MTP 模型 patch。

需要改造的部分：

- `_run_merged_draft` 不再在一个函数内同步完成 edge/cloud 往返；
- edge/cloud 通信由 Worker + SchedulerOutput 驱动；
- MTP_DRAFT 的输出写入 DraftTokenIds 缓冲，而不是直接返回给 `sample_tokens()`。

避免复用：

- 不复用普通 `ExecuteModelState` 表示 MTP_DRAFT；
- 不复用 prefill layerwise continuation state；
- 不复用 `DECODE_LAST` 的 sample path 生成 draft token ids。

---

# 13. 请求生命周期

MTP 开启后，一个请求稳定 decode 生命周期：

```text
running[]
  |
  | VERIFY_DECODE_FIRST
  v
verify_decode_inflight
  |
  | VERIFY_DECODE_LAST
  v
running[] 或 finished
  |
  | 若未 finished，创建 MTPDraftTask
  v
mtp_draft_waiting[]
  |
  | MTP_DRAFT_FIRST
  v
mtp_draft_inflight
  |
  | MTP_DRAFT_LAST
  v
draft tokens ready
  |
  | Scheduler.update_draft_token_ids
  v
下一轮 VERIFY_DECODE_FIRST 携带 scheduled_spec_decode_tokens
```

关键点：

```text
请求是否 finished 只由 VERIFY_DECODE_LAST 决定。
MTP_DRAFT 不改变请求 finished 状态。
```

如果请求在 MTP_DRAFT 执行期间被 abort：

```text
丢弃对应 draft task；
清理 MTPDraftState；
释放 MTP_DRAFT channel；
不得回填 draft token ids。
```

---

# 14. 正确性不变量

```text
1. VERIFY_DECODE_LAST 是唯一产生用户可见 token 的 decode tail。
2. MTP_DRAFT_LAST 只产生 draft token ids，不产生用户 token。
3. VERIFY_DECODE_LAST 不同步等待 MTP_DRAFT 完成。
4. MTP_DRAFT_FIRST/LAST 必须有独立 head_token。
5. MTP_DRAFT 必须携带 parent_head_token，关联触发它的 VERIFY_DECODE_LAST。
6. VERIFY_DECODE 和 MTP_DRAFT 不得共享同一个 HeadState。
7. VERIFY_DECODE 和 MTP_DRAFT 不得无保护复用同一个 hidden channel。
8. PREFILL_LAST / VERIFY_DECODE_LAST / MTP_DRAFT_LAST 均不得发往云侧。
9. PREFILL_FIRST / VERIFY_DECODE_FIRST / MTP_DRAFT_FIRST 才允许进入 PRE_OUT。
10. `prefill_inflight_count` 只由 P首/P尾修改。
11. `decode_inflight_count` 只由 VERIFY_D首/VERIFY_D尾修改。
12. `mtp_draft_inflight_count` 只由 MTP_DRAFT首/尾修改。
13. MTP_DRAFT 结果回填前必须校验请求仍存在且未 abort。
14. draft tokens 未 ready 时，VERIFY_DECODE 可以无 draft 运行，但不能切换到另一套普通 decode 状态机。
15. control-plane `head_token` 与 data-plane `_head_token` 必须一致。
```

---

# 15. 风险点

| 风险 | 说明 | 建议 |
|---|---|---|
| tail drain 被 MTP_DRAFT 阻塞 | 若 MTP_DRAFT 仍在 sample_tokens 内同步执行，会直接破坏 PD 掩盖 | 必须异步拆分 |
| channel 串包 | VERIFY_DECODE 和 MTP_DRAFT 共用 DECODE channel 容易错配 | 新增 `MTP_DRAFT` channel |
| draft result stale | 请求 abort/finished 后 MTP_DRAFT 才返回 | 回填前校验 req_id 和 generation/version |
| layerwise state 覆盖 | MTP_DRAFT中 与 P中 slice 交错时复用 `_layerwise_intermediate` | MTP_DRAFT 禁止切片或新增独立 state |
| 云侧饥饿 | P中/VERIFY_D中/MTP_DRAFT中 三类任务互相压制 | 三态期望调度 + fallback 保持状态 |
| 首轮无 draft | 刚进入 decode 时没有 scheduled_spec_decode_tokens | 允许 VERIFY_DECODE 无 draft 执行，不视为模式回退 |
| 多 draft step 延迟 | `num_speculative_tokens > 1` 时 MTP_DRAFT 完成时间变长 | 第一阶段按 step 拆小任务 |
| Graph capture stale metadata | MTP draft attention metadata 依赖 positions/seq_lens/slot_mapping | 复用来源项目刷新逻辑并按 draft_step_idx 校验 |
| EngineCore 回填时序 | draft tokens 可能晚于下一轮调度 | Scheduler 支持 draft miss，下一轮继续尝试 |
| 兼容旧 DECODE batch type | 目标代码已有 DECODE_FIRST/LAST 逻辑 | 先将其定义为 VERIFY_DECODE alias，MTP_DRAFT 独立新增 |

---

# 16. 分阶段实施建议

## 16.1 Phase M1：语义拆分，不改云侧三态

目标：

- `sample_tokens()` 不再同步执行 MTP proposer；
- 新增 `MTPDraftTask` 队列；
- `VERIFY_DECODE_LAST` 只 enqueue task；
- MTP_DRAFT 暂时同步本地执行，用于打通数据结构。

验收：

```text
用户 token 输出不变；
draft token ids 能通过 take_draft_token_ids 回填；
VERIFY_DECODE_LAST latency 不包含 MTP proposer 计算。
```

## 16.2 Phase M2：MTP_DRAFT 首尾分离

目标：

- 新增 `MTP_DRAFT_FIRST/LAST`；
- 新增 `mtp_drafts_last_ready[]`；
- 新增 `mtp_draft_inflight_count`；
- Worker 支持 MTP_DRAFT batch_type 派发；
- MTP_DRAFT 走 edge/cloud 分段。

验收：

```text
MTP_DRAFT 可异步跨边云执行；
VERIFY_DECODE_LAST 不等待 MTP_DRAFT_LAST；
draft tokens 可供下一轮 VERIFY_DECODE 使用。
```

## 16.3 Phase M3：云侧三态掩盖调度

目标：

- PassiveScheduler 新增 `ready_mtp_drafts[]`；
- 云侧状态扩展为 EEP/EEV/EEM；
- 支持 P中 / VERIFY_D中 / MTP_DRAFT中 交替和 fallback。

验收：

```text
云侧无明显 decode 间气泡；
P中 和 MTP_DRAFT中 可覆盖 VERIFY_D尾 + VERIFY_D首 + 通信时延；
无 tail ready 堆积。
```

## 16.4 Phase M4：多 draft step 与性能优化

目标：

- 支持 `num_speculative_tokens > 1`；
- 支持 MTP_DRAFT graph capture；
- 评估 MTP_DRAFT layerwise slicing；
- 引入 draft task aging，避免长期 draft miss。

---

# 17. 推荐配置

第一阶段推荐：

```python
pd_prefill_inflight_limit = 2
decode_inflight_limit = 1
mtp_draft_inflight_limit = 1
enable_mtp_draft_dedicated_channel = True
enable_mtp_draft_layerwise = False
cloud_dispatch_policy = "prefill_verify_mtp_expectation"
```

策略含义：

| 配置 | 含义 |
|---|---|
| `pd_prefill_inflight_limit = 2` | 保持 2P1D 掩盖能力 |
| `decode_inflight_limit = 1` | 主 verify decode 不并发，降低状态复杂度 |
| `mtp_draft_inflight_limit = 1` | MTP proposer 一次只跑一个 batch |
| `enable_mtp_draft_dedicated_channel = True` | VERIFY_DECODE 与 MTP_DRAFT 通道隔离 |
| `enable_mtp_draft_layerwise = False` | 第一阶段禁止 MTP_DRAFT 云侧切片 |

---

# 18. 总结

MTP 加入 PD 分离后，核心不是增加一套“MTP decode”主流程，而是把 decode 拆成：

```text
VERIFY_DECODE：主模型验证，产生用户 token
MTP_DRAFT：proposer 生成下一轮 draft token ids
```

`VERIFY_DECODE` 仍然是 Scheduler 的 decode 请求，只是携带 `scheduled_spec_decode_tokens`。
`MTP_DRAFT` 是由 `VERIFY_DECODE_LAST` 派生的异步任务，输出 draft token ids，不更新用户可见 token。

调度框架必须保证：

```text
P尾 / VERIFY_D尾 / MTP_DRAFT尾 优先 drain；
MTP_DRAFT 不阻塞 VERIFY_DECODE_LAST；
MTP_DRAFT 有独立状态、队列、inflight 计数和 hidden channel；
云侧在 P中 / VERIFY_D中 / MTP_DRAFT中 之间做期望型交替调度。
```

这样才能在保持 vllm-ascend speculative decoding 语义的同时，继续实现 PD 掩盖流水线。
