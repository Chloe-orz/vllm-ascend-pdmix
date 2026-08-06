# Claude Code Session

> **Resume in CLI:** `claude --resume 405d0614-8335-415b-86ba-428e103bbad7`

| Field | Value |
|---|---|
| **Project** | `c:\theroadtofree-github` |
| **Session ID** | `405d0614-8335-415b-86ba-428e103bbad7` |
| **Working Dir** | `c:\theroadtofree_github` |
| **Started** | 2026/8/5 20:49:21 |
| **Last Updated** | 2026/8/6 09:26:57 |
| **Messages** | 235 |

---

## User <sup>2026/8/5 20:49:21</sup>

<ide_opened_file>The user opened the file c:\theroadtofree_github\vllm\vllm\v1\executor\multiproc_executor.py in the IDE. This may or may not be related to the current task.</ide_opened_file>

非边云的dp并行场景，两台服务器，每个服务器8卡，服务器0是dp0，服务器1是dp1，kimi25_w4a8_static_m6模型，单请求就出现挂死，挂死的栈信息和日志信息在C:\theroadtofree_github\log\0805-1.txt；问题应该本工程C:\theroadtofree_github\vllm新增的边云dp并行场景修改导致的，代码可以和C:\yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy目录比较


---

## Assistant <sup>2026/8/5 20:49:34</sup>


---

## Assistant <sup>2026/8/5 20:49:34</sup>

我先分析这个问题。让我并行读取日志文件和了解代码结构。


---

## Assistant <sup>2026/8/5 20:49:34</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\theroadtofree_github\\log\\0805-1.txt"
}
```

</details>


---

## User <sup>2026/8/5 20:49:34</sup>

<details>
<summary>Tool Result</summary>

```
1	1、dp0挂死栈，其他tp和tp0一样
2	CMD: VLLM::Worker_DP0_TP0_EP0
3	--------------------------------------------------------------------------------
4	MainThread call stack:
5	Thread 698984 (idle): "MainThread"
6	    read (libc.so.6)
7	    eventfd_read (libc.so.6)
8	    c10_npu::NPUEvent::record (libtorch_npu.so)
9	    c10d_npu::ProcessGroupHCCL::_reduce_scatter_base (libtorch_npu.so)
10	    c10d::ops::(anonymous namespace)::_reduce_scatter_base_PrivateUse1 (libtorch_cpu.so)
11	    c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<std::tuple<at::Tensor, c10::intrusive_ptr<c10d::Work, c10::detail::intrusive_target_default_null_type<c10d::Work> > > (*)(at::Tensor&, at::Tensor&, c10::intrusive_ptr<c10d::ProcessGroup, c10::detail::intrusive_target_default_null_type<c10d::ProcessGroup> > const&, c10::intrusive_ptr<c10d::ReduceOp, c10::detail::intrusive_target_default_null_type<c10d::ReduceOp> > const&, bool, long), std::tuple<at::Tensor, c10::intrusive_ptr<c10d:
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:49:34</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\Users\\root\\.claude\\projects\\c--theroadtofree-github\\memory\\dp-segment-coordination.md"
}
```

</details>


---

## User <sup>2026/8/5 20:49:34</sup>

<details>
<summary>Tool Result</summary>

```
1	---
2	name: dp-segment-coordination
3	description: Edge-cloud DP parallel prefill optimization - edge-only prefill-defer to avoid one-real-one-dummy
4	metadata: 
5	  node_type: memory
6	  type: project
7	  originSessionId: 8ffbc28b-780a-49d1-98f5-451be8e207f5
8	  modified: 2026-08-04T04:52:14.446Z
9	---
10	
11	边云 DP 并行（全 MoE + EP，dp=2）下，prefill 阶段两 DP 是「一个 real prefill + 一个 dummy」，decode 已优化为两 real。根因：边侧 `_coordinate_bt` / 云侧 `_coordinate_decision` 每 tick 选单一 winner batch_type 强制所有 DP 同类型，无该类型工作的 DP 产 dummy。
12	
13	**用户确认的关键约束（非代码可推导）**：
14	- 全 MoE：边侧 head 段（P首/D首）、tail 段（P尾/D尾）也走跨 DP EP all-toall。head 层 ≠ tail 层，跨段不能同 tick 都 real。
15	- EP all-toall 按 routing 逐 token 收发，不要求同 batch_type，只要求同层同段。PF+DF（都 head）/ PL+DL（都 tail）可同 tick 两 real。
16	- 云侧恒 middle，PF中+D中 任意混都兼容。
17	
18	**尝试过的方案（已回退）**：段级协调 Phase1（边云两侧改，混跑 PF+DF）。用户实测挂死，已完全回退。挂死原因疑似：云侧放开 sync_queue_state + 混跑后两 cloud DP 错拍致 sync_metadata 死锁，或边侧段对齐时序问题。教训：**不要动云侧**。
19	
20	**当前方案（简单，仅边侧）**：prefill-defer。开关 `additional_config.edge_
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:50:10</sup>


---

## Assistant <sup>2026/8/5 20:50:14</sup>

这提供了很多信息。在深入研究之前，先让我分析一下挂死的症状。

**挂死症状分析：**
- **dp0** (dp_rank=0)：卡在 `reduce_scatter`（TP 通信）中，位于实际 `pd_mix` 前向传播期间（tokens=13）
- **dp1** (dp_rank=1)：卡在 `_sync_metadata_across_dp` 的 `all_reduce` 中，位于 `execute_dummy_batch` -> `_dummy_run` 期间，日志显示 `role=cloud peer_bt=0 skip_head=True skip_tail=True`

**关键观察结果：** 这是一个**非边云 DP** 场景（只有两台普通服务器，dp=2），但 dp1 的日志显示了边云特有的逻辑（`role=cloud`，`skip_head/skip_tail`，重复的 dummy 批次）。边云的修改正在非边云场景中被触发。dp0 和 dp1 处于完全不同的状态（dp0 为真实工作，dp1 为一连串的 dummy 运行）→ 跨 DP 的集合通信计数不匹配 → 死锁。

让我探索代码结构和关键文件以确认根本原因。


---

## Assistant <sup>2026/8/5 20:50:15</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "ls -la && echo \"---VLLM---\" && ls vllm/ 2>/dev/null | head -20 && echo \"---vllm_ascend search---\" && find . -maxdepth 4 -type d -name \"vllm_ascend\" 2>/dev/null && echo \"---reference dir---\" && ls \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\" 2>/dev/null | head -30",
  "description": "Explore project and reference directory structure"
}
```

</details>


---

## Assistant <sup>2026/8/5 20:50:17</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "echo \"=== current multiproc_executor.py lines 440-470, 1255-1310 ===\" && sed -n '440,470p;1255,1310p' \"c:/theroadtofree_github/vllm/vllm/v1/executor/multiproc_executor.py\" 2>/dev/null",
  "description": "Read key multiproc_executor.py sections"
}
```

</details>


---

## User <sup>2026/8/5 20:50:24</sup>

<details>
<summary>Tool Result</summary>

```
=== current multiproc_executor.py lines 440-470, 1255-1310 ===
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)

        # [EDGE-ENQUEUE] 参考 passive_core.py:590 格式记录边侧 enqueue 耗时
        _bt = "N/A"
        if (
            isinstance(method, str)
            and method == "execute_model"
            and args
            and hasattr(args[0], "batch_type")
            and args[0].batch_type is not None
        ):
            _bt = args[0].batch_type.value
        _t0 = time.monotonic()
        self.rpc_broadcast_mq.enqueue(
            (send_method, args, kwargs, output_rank), local_only=local_only
        )
        _dt_ms = (time.monotonic() - _t0) * 1000
        logger.info(
            "[EDGE-ENQUEUE] %s enqueue took %.3f ms",
            _bt,
            _dt_ms,
        )

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        def get_re
...(truncated)
```

</details>


---

## User <sup>2026/8/5 20:50:25</sup>

<details>
<summary>Tool Result</summary>

```
total 68
drwxr-xr-x 1 root 197121 0 Jul 30 23:24 .
drwxr-xr-x 1 root 197121 0 Aug  5 11:25 ..
drwxr-xr-x 1 root 197121 0 Aug  4 21:31 .claude
drwxr-xr-x 1 root 197121 0 Aug  5 20:50 .claude-code-history
drwxr-xr-x 1 root 197121 0 Aug  5 20:44 log
drwxr-xr-x 1 root 197121 0 Jul 29 22:06 vllm
drwxr-xr-x 1 root 197121 0 Aug  3 10:53 vllm-ascend
---VLLM---
AGENTS.md
CLAUDE.md
CMakeLists.txt
CODE_OF_CONDUCT.md
CONTRIBUTING.md
DCO
LICENSE
MANIFEST.in
README.md
RELEASE.md
SECURITY.md
benchmarks
cmake
codecov.yml
csrc
docker
docs
examples
mkdocs.yaml
pyproject.toml
---vllm_ascend search---
./vllm-ascend/vllm_ascend
---reference dir---
vllm
vllm-ascend
```

</details>


---

## Assistant <sup>2026/8/5 20:50:30</sup>


---

## Assistant <sup>2026/8/5 20:50:31</sup>

参考目录 `C:\yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy` 中包含 `vllm` 和 `vllm-ascend`。让我找到关键的协调逻辑，并将其与参考目录进行比较。我将启动并行 Agent 来调查不同方面。


---

## Assistant <sup>2026/8/5 20:50:33</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== patch_engine_core.py location ===\" && find . -name \"patch_engine_core.py\" 2>/dev/null && echo \"=== ref ===\" && find \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\" -name \"patch_engine_core.py\" 2>/dev/null && echo \"=== pd_separated_scheduler.py ===\" && find . -name \"pd_separated_scheduler.py\" 2>/dev/null && find \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\" -name \"pd_separated_scheduler.py\" 2>/dev/null",
  "description": "Locate key coordination files in both trees"
}
```

</details>


---

## Assistant <sup>2026/8/5 20:50:34</sup>

<details>
<summary>Tool: <code>Grep</code></summary>

```json
{
  "-n": true,
  "head_limit": 60,
  "output_mode": "content",
  "path": "c:/theroadtofree_github/vllm-ascend",
  "pattern": "_coordinate_bt|_coordinate_decision|sync_metadata_across_dp|role=cloud|skip_head|edge_cloud_config|is_edge_cloud|dp_prefill_defer"
}
```

</details>


---

## User <sup>2026/8/5 20:50:34</sup>

<details>
<summary>Tool Result</summary>

```
vllm-ascend\docs\PDbatch分离边云协同Phase2-4 extend详细设计.md:560:    if not self._is_edge_cloud:
vllm-ascend\docs\PDbatch分离边云协同Phase2-4 extend详细设计.md:715:    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"cloud",...}}' \
vllm-ascend\docs\PDbatch分离边云协同Phase2-4 extend详细设计.md:723:    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge",...}}' \
vllm-ascend\docs\PDbatch分离边云协同Phase1详细设计.md:48:| `edge_cloud_config` | dict | `additional_config` | 昇腾专属：层切分策略、角色、模式等 |
vllm-ascend\docs\PDbatch分离边云协同Phase1详细设计.md:98:    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","edge_head_tail_layers":1}}'
vllm-ascend\docs\PDbatch分离边云协同Phase1详细设计.md:104:    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","edge_head_tail_layers":1}}'
vllm-ascend\docs\PDbatch分离边云协同Phase1详细设计.md:171:--additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","edge_head_tail_layers":1}}'
vllm-ascend\docs\PDbatch分离边云协同Phase1详细设计.md:175:--ad
...(truncated)
```

</details>


---

## User <sup>2026/8/5 20:50:35</sup>

<details>
<summary>Tool Result</summary>

```
=== patch_engine_core.py location ===
./vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py
=== ref ===
C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py
=== pd_separated_scheduler.py ===
./vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py
C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py
```

</details>


---

## Assistant <sup>2026/8/5 20:50:43</sup>


---

## Assistant <sup>2026/8/5 20:50:44</sup>

我来读取关键协调逻辑。日志显示 dp1 正在接收虚拟批次（`[EDGE-ENQUEUE] N/A`）并被分配为 `role=cloud` —— 这在非边云场景中是不应该发生的。我来查找核心协调函数。


---

## Assistant <sup>2026/8/5 20:50:45</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github/vllm-ascend/vllm_ascend\" && echo \"=== grep _coordinate in patch_engine_core ===\" && grep -n \"_coordinate\\|def _maybe\\|is_pd_dummy\\|dummy\\|enqueue.*dummy\\|_determine_winner\\|EMPTY\\|batch_type\" patch/platform/patch_engine_core.py | head -80",
  "description": "Find coordination functions in patch_engine_core.py"
}
```

</details>


---

## User <sup>2026/8/5 20:50:46</sup>

<details>
<summary>Tool Result</summary>

```
=== grep _coordinate in patch_engine_core ===
75:from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
219:        bt = so.batch_type
220:        logger.info(f"Received scheduler_output from cloud, batch_type: {bt}")
237:                "PD-separation POST_OUT received unexpected batch_type=%s; "
244:def _maybe_publish_pre_out(
250:    bt = scheduler_output.batch_type
284:        BatchType.EMPTY,
292:            "PD-separation PRE_OUT skipping non-separated batch_type=%s",
336:    if scheduler_output.batch_type not in (
347:    """Stamp the edge-side original_seq on an EngineCore-created dummy.
352:    ``_patched_execute_dummy_batch`` / ``_publish_pd_dummy_zmq``) bypass
356:    log never shows a dummy with original_seq=None.
373:    bt = scheduler_output.batch_type
378:    """Keep worker-side cleanup from EMPTY batches for the next real batch."""
398:    """Attach cleanup skipped with EMPTY batches to the next worker batch."""
418:    """Complete an EMPTY SchedulerOu
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:50:47</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github/vllm-ascend/vllm_ascend\" && echo \"=== model_runner _sync_metadata + dummy + role/skip ===\" && grep -n \"def _sync_metadata_across_dp\\|def _dummy_run\\|def _determine_batch_execution\\|role=cloud\\|role.*cloud\\|is_edge_device\\|def is_edge\\|skip_head\\|skip_tail\\|_is_edge_cloud\\|is_pd_dummy\\|def execute_model\\|execute_dummy_batch\\|all_reduce\" worker/model_runner_v1.py | head -80",
  "description": "Find dummy/sync/role functions in model_runner_v1.py"
}
```

</details>


---

## User <sup>2026/8/5 20:50:48</sup>

<details>
<summary>Tool Result</summary>

```
=== model_runner _sync_metadata + dummy + role/skip ===
52:    is_edge_device,
525:        # 2=PL, 3=DF, 4=DL). Packed into sync_metadata all_reduce so the
590:        self._is_edge_cloud_tail_segment = False
597:            expected_role = "edge" if self.parallel_config.is_edge_node else "cloud"
602:                    f"{expected_role!r}, got {self.edge_cloud_cfg.role!r}."
936:            expected_role = "edge" if self.parallel_config.is_edge_node else "cloud"
941:                    f"{expected_role!r}, got {self.edge_cloud_cfg.role!r}."
1292:        #    → 根据 is_edge_device() 计算本侧本地层索引
1539:        if is_edge_device():
1562:        if not is_edge_device():
1669:    def _sync_metadata_across_dp(
1678:        # include them in the all_reduce operation, and more over, we CANNOT skip it
1693:        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)
1700:        # all_reduce is a barrier so call N on both sides must pair. my_bt=0
2108:        if not self._is_edge_cloud_tail
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:50:58</sup>


---

## Assistant <sup>2026/8/5 20:50:58</sup>

这是一个复杂的代码库。让我直接读取关键的门控函数。关键问题是 `_is_coordinated_dp()`（以及虚拟批处理机制）在非边云场景下是否被错误地激活了。


---

## Assistant <sup>2026/8/5 20:50:59</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py",
  "limit": 200,
  "offset": 680
}
```

</details>


---

## User <sup>2026/8/5 20:50:59</sup>

<details>
<summary>Tool Result</summary>

```
680	# =======================================================================#
681	# EngineCore.step — full replacement, mirrors upstream + dest inserts.    #
682	# =======================================================================#
683	# batch_type id packed into the coordination all_reduce. 0 = EMPTY/idle,
684	# 1/2/3/4 = real PF/PL/DF/DL. A dummy SchedulerOutput carries the WINNER bt
685	# (NOT 0): both DPs contribute the same winner id, so peer_bt != 0 and the
686	# _dummy_run skip-both branch (peer_bt == 0) never fires under coordination
687	# - this is what kills the old self-driven-dummy count-drift deadlock path.
688	_BT_COORD_ID = {
689	    BatchType.EMPTY: 0,
690	    BatchType.PREFILL_FIRST: 1,
691	    BatchType.PREFILL_LAST: 2,
692	    BatchType.DECODE_FIRST: 3,
693	    BatchType.DECODE_LAST: 4,
694	}
695	_BT_COORD_ID_INV = {v: k for k, v in _BT_COORD_ID.items()}
696	
697	
698	def _is_coordinated_dp(self) -> bool:
699	    """True when edge-side cross-DP batch_type coord
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:52:05</sup>


---

## Assistant <sup>2026/8/5 20:52:08</sup>

现在我发现了一个关键线索。`_is_coordinated_dp()` 的定义如下：

```python
def _is_coordinated_dp(self) -> bool:
    return (
        getattr(self, "dp_group", None) is not None       # dp>1
        and getattr(self, "_pp_pd_channel", None) is not None  # PD-separation
        and bool(getattr(self.vllm_config.model_config, "is_moe", False))  # MoE
    )
```

**该门控机制并没有检查 `edge_cloud_config.enabled` 或 `is_edge_device()`。** 它只检查了 dp>1、PD-separation 通道以及 MoE 模型——在非边云的 PD-separation MoE 场景下，这三者均可能为真。这将会触发跨 DP 的 dummy-batch 协调，并导致 dp1 上的 `role=cloud`/`skip_head/skip_tail` 代码路径被执行。

让我来验证一下参考目录的性质，并收集关于门控和 dummy 路径的关键事实。


---

## Assistant <sup>2026/8/5 20:52:10</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== diff stat: current vs reference (vllm-ascend key files) ===\" && for f in vllm_ascend/patch/platform/patch_engine_core.py vllm_ascend/worker/model_runner_v1.py vllm_ascend/worker/worker.py vllm_ascend/core/pd_separated_scheduler.py; do echo \"--- $f ---\"; diff \"vllm-ascend/$f\" \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/$f\" >/dev/null 2>&1 && echo \"IDENTICAL\" || echo \"DIFFERS ($( diff \"vllm-ascend/$f\" \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/$f\" 2>/dev/null | grep -c '^[<>]' ) changed lines)\"; done",
  "description": "Compare key files current vs reference"
}
```

</details>


---

## Assistant <sup>2026/8/5 20:52:13</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== is_edge_device def ===\" && grep -rn \"def is_edge_device\\|def is_edge_cloud_pp_mode\\|_edge_cloud_enabled =\" vllm-ascend/vllm_ascend/ | head -20 && echo \"=== does reference have _is_coordinated_dp / _coordinate_bt? ===\" && grep -cn \"_is_coordinated_dp\\|_coordinate_bt\\|_coord_bt_count\" \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py\" && echo \"=== reference _sync_metadata / dummy ===\" && grep -cn \"_sync_metadata_across_dp\\|_dummy_run\\|skip_head\" \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/worker/model_runner_v1.py\"",
  "description": "Check edge device gating and reference coordination presence"
}
```

</details>


---

## User <sup>2026/8/5 20:55:13</sup>

<details>
<summary>Tool Result</summary>

```
=== diff stat: current vs reference (vllm-ascend key files) ===
--- vllm_ascend/patch/platform/patch_engine_core.py ---
DIFFERS (0 changed lines)
--- vllm_ascend/worker/model_runner_v1.py ---
DIFFERS (0 changed lines)
--- vllm_ascend/worker/worker.py ---
DIFFERS (0 changed lines)
--- vllm_ascend/core/pd_separated_scheduler.py ---
DIFFERS (0 changed lines)
```

</details>


---

## User <sup>2026/8/5 20:55:14</sup>

<details>
<summary>Tool Result</summary>

```
=== is_edge_device def ===
vllm-ascend/vllm_ascend/worker/model_runner_v1.py:586:        self._edge_cloud_enabled = self.edge_cloud_cfg.enabled
vllm-ascend/vllm_ascend/worker/model_runner_v1.py:929:        self._edge_cloud_enabled = self.edge_cloud_cfg.enabled
=== does reference have _is_coordinated_dp / _coordinate_bt? ===
0
```

</details>


---

## Assistant <sup>2026/8/5 20:55:24</sup>


---

## Assistant <sup>2026/8/5 20:55:26</sup>

对比结果不明确（"DIFFERS (0 changed lines)" 很可疑——可能是换行符问题）。但已确认：**参考代码中没有 `_is_coordinated_dp`/`_coordinate_bt`** —— 这个协调逻辑是为了边云 DP 新增的。我来获取更清晰的对比结果并找到 `is_edge_device` 的定义。


---

## Assistant <sup>2026/8/5 20:55:29</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== file sizes & line counts ===\" && for f in vllm_ascend/patch/platform/patch_engine_core.py vllm_ascend/worker/model_runner_v1.py; do echo \"--- $f ---\"; wc -l \"vllm-ascend/$f\" \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/$f\" 2>&1; done && echo \"=== line ending check ===\" && file \"vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py\" \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py\" 2>&1 && echo \"=== normalized diff line count (ignore CR) ===\" && diff <(tr -d '\\r' < \"vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py\") <(tr -d '\\r' < \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py\") | grep -c '^[<>]' && echo \"=== ref model_runner has sync_meta/dummy? ===\" && grep -cn \"_sync_metadata_across_dp\\|_dummy_run\\|skip_head\\|_is_coordinated\\|edge_cloud\" \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/worker/model_runner_v1.py\"",
  "description": "Clean normalized diff and line ending check"
}
```

</details>


---

## Assistant <sup>2026/8/5 20:55:31</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== where is is_edge_device defined ===\" && grep -rn \"def is_edge_device\\|def is_edge_cloud_pp_mode\\|def is_edge_node\" vllm-ascend/vllm_ascend/ && echo \"=== is_edge_device import in model_runner ===\" && grep -n \"is_edge_device\\|is_edge_cloud_pp_mode\\|is_edge_node\" vllm-ascend/vllm_ascend/worker/model_runner_v1.py | head -10",
  "description": "Find is_edge_device definition"
}
```

</details>


---

## User <sup>2026/8/5 20:55:33</sup>

<details>
<summary>Tool Result</summary>

```
=== file sizes & line counts ===
--- vllm_ascend/patch/platform/patch_engine_core.py ---
 1519 vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py
wc: 'C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm_ascend/patch/platform/patch_engine_core.py': No such file or directory
 1519 total
--- vllm_ascend/worker/model_runner_v1.py ---
  9333 vllm-ascend/vllm_ascend/worker/model_runner_v1.py
wc: 'C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm_ascend/worker/model_runner_v1.py': No such file or directory
  9333 total
```

</details>


---

## User <sup>2026/8/5 20:55:34</sup>

<details>
<summary>Tool Result</summary>

```
Exit code 1
=== where is is_edge_device defined ===
```

</details>


---

## Assistant <sup>2026/8/5 20:55:38</sup>


---

## Assistant <sup>2026/8/5 20:55:38</sup>

路径问题 - 存在嵌套的 `vllm_ascend` 目录。我来修复一下并找到 `is_edge_device`。


---

## Assistant <sup>2026/8/5 20:55:41</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && BASE_CUR=\"vllm-ascend/vllm_ascend\" && BASE_REF=\"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend\" && echo \"=== normalized diff line counts (ignore CR) ===\" && for f in patch/platform/patch_engine_core.py worker/model_runner_v1.py worker/worker.py core/pd_separated_scheduler.py; do n=$(diff <(tr -d '\\r' < \"$BASE_CUR/$f\") <(tr -d '\\r' < \"$BASE_REF/$f\") 2>/dev/null | grep -c '^[<>]'); echo \"$f : $n changed lines\"; done && echo \"=== ref model_runner edge/dummy/sync presence ===\" && grep -c \"_sync_metadata_across_dp\\|_dummy_run\\|skip_head\\|edge_cloud\" \"$BASE_REF/worker/model_runner_v1.py\"",
  "description": "Corrected normalized diff line counts"
}
```

</details>


---

## Assistant <sup>2026/8/5 20:55:43</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== is_edge_device def (broad) ===\" && grep -rn \"def is_edge_device\\|def is_edge_cloud_pp_mode\\|def is_edge_node\\|def is_edge_device(\" vllm-ascend/ 2>/dev/null | head && echo \"=== search whole vllm-ascend ===\" && grep -rln \"is_edge_device\" vllm-ascend/vllm_ascend/ | head -20",
  "description": "Find is_edge_device definition broadly"
}
```

</details>


---

## User <sup>2026/8/5 20:55:48</sup>

<details>
<summary>Tool Result</summary>

```
=== normalized diff line counts (ignore CR) ===
patch/platform/patch_engine_core.py : 648 changed lines
worker/model_runner_v1.py : 503 changed lines
worker/worker.py : 173 changed lines
core/pd_separated_scheduler.py : 156 changed lines
=== ref model_runner edge/dummy/sync presence ===
235
```

</details>


---

## User <sup>2026/8/5 20:55:50</sup>

<details>
<summary>Tool Result</summary>

```
=== is_edge_device def (broad) ===
vllm-ascend/docs/PDbatch分离边云协同Phase1详细设计.md:457:def is_edge_device() -> bool:
=== search whole vllm-ascend ===
vllm-ascend/vllm_ascend/distributed/parallel_state.py
vllm-ascend/vllm_ascend/distributed/__pycache__/parallel_state.cpython-314.pyc
vllm-ascend/vllm_ascend/worker/model_runner_v1.py
vllm-ascend/vllm_ascend/worker/worker.py
vllm-ascend/vllm_ascend/worker/__pycache__/model_runner_v1.cpython-314.pyc
vllm-ascend/vllm_ascend/worker/__pycache__/worker.cpython-314.pyc
```

</details>


---

## Assistant <sup>2026/8/5 20:56:15</sup>


---

## Assistant <sup>2026/8/5 20:56:16</sup>

参考工程中包含了边云相关的代码，但**没有**协调逻辑（`_is_coordinated_dp`/`_coordinate_bt`）—— 这是为了边云 DP 新增的。变更规模：patch_engine_core.py (+648 行)，model_runner_v1.py (+503 行)。`is_edge_device` 定义在 `vllm_ascend/distributed/parallel_state.py` 中。

我来并行读取一下关键函数，以验证该假设。


---

## Assistant <sup>2026/8/5 20:56:17</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
  "limit": 70,
  "offset": 1669
}
```

</details>


---

## User <sup>2026/8/5 20:56:17</sup>

<details>
<summary>Tool Result</summary>

```
1669	    def _sync_metadata_across_dp(
1670	        self,
1671	        num_tokens: int,
1672	        is_draft_model: bool = False,
1673	        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
1674	        allow_dp_padding: bool = False,
1675	    ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
1676	        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
1677	        # our case, we still need to sync the other two flags as well. So we need to
1678	        # include them in the all_reduce operation, and more over, we CANNOT skip it
1679	        # even if we are running in eager mode, which harms performance.
1680	        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
1681	        # immediately once the other two flags are no longer needed.
1682	        if self.dp_size == 1:
1683	            return num_tokens, None, cudagraph_mode
1684	
1685	        if should_skip_allreduce_across_dp_group(self.vllm_config, is_draft_model):
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:56:17</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
  "limit": 180,
  "offset": 7640
}
```

</details>


---

## User <sup>2026/8/5 20:56:17</sup>

<details>
<summary>Tool Result</summary>

```
7640	            dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)
7641	
7642	            def dummy_compute_logits(hidden_states):
7643	                if not need_dummy_logits:
7644	                    return None
7645	                return self.model.compute_logits(hidden_states[dummy_indices])
7646	
7647	            def dummy_drafter_compute_logits(hidden_states):
7648	                if not need_dummy_logits or self.drafter is None:
7649	                    return
7650	                if hasattr(self.drafter, "model") and hasattr(self.drafter.model, "compute_logits"):
7651	                    return self.drafter.model.compute_logits(hidden_states[dummy_indices])
7652	
7653	            # PD-separation: DUMMY matches real DP's segment.
7654	            # - peer_bt == 0 (both DUMMY / idle): skip both segments for BOTH
7655	            #   edge and cloud. No real data to all_gather, so no MoE forward
7656	            #   and no cross-DP EP all-toall; only sync_met
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:56:18</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github/vllm-ascend/vllm_ascend\" && echo \"=== is_edge_device / is_edge_cloud_pp_mode / is_edge_node in parallel_state.py ===\" && grep -n \"def is_edge_device\\|def is_edge_cloud_pp_mode\\|def is_edge_node\\|_EDGE_DEVICE\\|edge_node\\|edge_cloud\" distributed/parallel_state.py | head -40",
  "description": "Find edge device gating in parallel_state"
}
```

</details>


---

## User <sup>2026/8/5 20:56:20</sup>

<details>
<summary>Tool Result</summary>

```
=== is_edge_device / is_edge_cloud_pp_mode / is_edge_node in parallel_state.py ===
60:# pair, atb_llm/.../edge_cloud_data_comm.py).
189:    # P2P protocol RTTs.  When True, edge_cloud_isend_tensor_dict concatenates
190:    # all tensors along dim=-1 before sending, and edge_cloud_irecv_tensor_dict
288:def _build_edge_cloud_tensor_meta(
386:def init_edge_cloud_tensor_meta(
396:    edge_cloud_irecv_tensor_dict to allocate receive buffers without
440:    _EDGE_CLOUD_TENSOR_META_E2C = _build_edge_cloud_tensor_meta(
443:    _EDGE_CLOUD_TENSOR_META_C2E = _build_edge_cloud_tensor_meta(
467:def get_edge_cloud_tensor_meta(
482:        "Call init_edge_cloud_tensor_meta() first."
487:def _select_edge_cloud_meta_for_send() -> EdgeCloudTensorMeta:
494:    return get_edge_cloud_tensor_meta(
499:def _select_edge_cloud_meta_for_recv() -> EdgeCloudTensorMeta:
508:    return get_edge_cloud_tensor_meta(
513:def warmup_edge_cloud_hidden_channels(
670:    if parallel_config.enable_edge_cloud:
742:         
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:56:40</sup>


---

## Assistant <sup>2026/8/5 20:56:42</sup>

关键发现：`_sync_metadata_across_dp` (第 1692 行) 报告了 `self._dp_batch_type_id`，并且日志显示 **dp0 (真实 pd_mix) 报告 `my_bt=0`**，而 dp1 (虚拟) 报告 `peer_bt=0` -> 触发了 `skip_head=skip_tail=True`。因此，pd_mix 被映射为了 batch_type id 0 (EMPTY)。这是核心不匹配所在。让我找到 `_dp_batch_type_id` 是在哪里设置的，以及 "pd_mix" 批处理类型是什么，并阅读 busy-loop 协调逻辑。


---

## Assistant <sup>2026/8/5 20:56:44</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github/vllm-ascend/vllm_ascend\" && echo \"=== _dp_batch_type_id assignments ===\" && grep -n \"_dp_batch_type_id\" worker/model_runner_v1.py && echo \"=== pd_mix batch type ===\" && grep -rn \"pd_mix\\|PDMix\\|PD_MIX\\|class BatchType\\|PREFILL_FIRST\\|DECODE_FIRST\" worker/model_runner_v1.py | head -20 && echo \"=== BatchType enum def location ===\" && grep -rln \"class BatchType\" . 2>/dev/null && grep -rln \"pd_mix\" vllm/ 2>/dev/null | head; grep -rln \"pd_mix\" \"c:/theroadtofree_github/vllm/vllm\" 2>/dev/null | head",
  "description": "Find _dp_batch_type_id and pd_mix batch type"
}
```

</details>


---

## Assistant <sup>2026/8/5 20:56:45</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py",
  "limit": 220,
  "offset": 1000
}
```

</details>


---

## User <sup>2026/8/5 20:56:45</sup>

<details>
<summary>Tool Result</summary>

```
1000	                ):
1001	                    return None, True
1002	                break
1003	
1004	        if not batch_queue:
1005	            # No completed/in-flight batch is available to collect. This can
1006	            # happen while waiting for a remote prefill tail.
1007	            return None, model_executed
1008	
1009	    else:
1010	        model_executed = False
1011	        deferred_scheduler_output = None
1012	        # [ascend insert] Pull cloud-returned tail-segment batches into the
1013	        # scheduler ready queues BEFORE coord/intent. This MUST run every step
1014	        # (unconditional, not gated on _should_schedule): when the edge is
1015	        # waiting for a cloud tail, prefills_last_ready is empty until drained,
1016	        # so _intended_batch_type() returns EMPTY -> winner EMPTY ->
1017	        # _should_schedule False. Gating the drain on _should_schedule would
1018	        # skip it forever -> PL/DL never drained -> prefill/decode never
1019	 
...(truncated)
```

</details>


---

## User <sup>2026/8/5 20:56:46</sup>

<details>
<summary>Tool Result</summary>

```
=== _dp_batch_type_id assignments ===
527:        self._dp_batch_type_id = 0
1692:        packed_tensor[2][self.dp_rank] = self._dp_batch_type_id
1712:            _sm_rank, self.dp_rank, _smc, self._dp_batch_type_id,
3563:        self._dp_batch_type_id = _bt_map.get(scheduler_output.batch_type, 0)
6040:        # PD-separation: set _dp_batch_type_id BEFORE _run_input_preparation
6044:        # before execute_model sets _dp_batch_type_id (line ~2416), so it uses
6054:        self._dp_batch_type_id = _bt_map.get(scheduler_output.batch_type, 0)
7412:        self._dp_batch_type_id = 0  # DUMMY
=== pd_mix batch type ===
574:        # request's PREFILL_FIRST ran between this request's PREFILL_LAST
575:        # and its first DECODE_FIRST).  Rows are views into the producing
2841:        1. Another request's batch (e.g. a PREFILL_FIRST) executed between
2842:           this request's PREFILL_LAST and its first DECODE_FIRST, wiping
3118:        # later verify batch still needs (its DECODE_FIRST
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:57:38</sup>


---

## Assistant <sup>2026/8/5 20:57:41</sup>

现在情况逐渐清晰了。关键事实如下：
- `_dp_batch_type_id = _bt_map.get(scheduler_output.batch_type, 0)` (第 3563 行) —— 如果 batch_type 不在 map 中，则默认为 **0 (EMPTY)**。该 map 仅包含 EMPTY/PF/PL/DF/DL。
- 第 7412 行：dummy 设置 `_dp_batch_type_id = 0`。
- 当 `_coordinated` 为 True 时，busy-loop 的协调机制会激活，并在空闲的 DP 上强制调度获胜者（产生 dummy）。

关键问题在于：**"pd_mix" 是什么**，以及 **`_pp_pd_channel` / 协调机制是否仅在边云模式下设置，还是在非边云的 PD-separation 中也会设置**？让我来验证一下这些门控机制。


---

## Assistant <sup>2026/8/5 20:57:42</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== BatchType enum ===\" && grep -n \"class BatchType\\|pd_mix\\|PREFILL_FIRST\\|DECODE_FIRST\\|DRAFT\\|EMPTY\\|PD_MIX\\|= \\\"\" vllm/vllm/v1/core/sched/output.py | head -40",
  "description": "Check BatchType enum for pd_mix"
}
```

</details>


---

## User <sup>2026/8/5 20:57:43</sup>

<details>
<summary>Tool Result</summary>

```
=== BatchType enum ===
141:class BatchType(enum.Enum):
148:    - PD_MIX:        prefill and decode requests in the same batch (default
154:    - EMPTY:         no tokens scheduled this step (sync-only batch, e.g.
156:    - PREFILL_FIRST: edge-cloud PD separation — prefill batch executing the
160:    - DECODE_FIRST:  edge-cloud PD separation — decode batch executing the
164:    - DRAFT_FIRST:   edge-cloud speculative draft batch executing the edge
166:    - DRAFT_LAST:    edge-cloud speculative draft batch executing the edge
169:    PD_MIX = "pd_mix"
170:    PURE_PREFILL = "pure_prefill"
171:    PURE_DECODE = "pure_decode"
172:    EMPTY = "empty"
174:    PREFILL_FIRST = "prefill_first"
175:    PREFILL_LAST = "prefill_last"
176:    DECODE_FIRST = "decode_first"
177:    DECODE_LAST = "decode_last"
178:    DRAFT_FIRST = "draft_first"
179:    DRAFT_LAST = "draft_last"
399:    # state. The base PD-mix Scheduler leaves it at the default (PD_MIX);
400:    # PDSeparatedScheduler overrides it to
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:57:45</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github/vllm-ascend/vllm_ascend\" && echo \"=== _pp_pd_channel assignment (current) ===\" && grep -rn \"_pp_pd_channel\\s*=\\|_pp_pd_channel =\" patch/platform/patch_engine_core.py | head && echo \"=== enable_edge_cloud flag ===\" && grep -rn \"enable_edge_cloud\" vllm/ 2>/dev/null | head; grep -rn \"enable_edge_cloud\" \"c:/theroadtofree_github/vllm/vllm\" 2>/dev/null | head -5; echo \"=== is_edge_device def in parallel_state ===\" && grep -n \"def is_edge_device\\|def is_edge_cloud_pp_mode\\|def is_edge_node\\|enable_edge_cloud\" distributed/parallel_state.py | head && echo \"=== execute_dummy_batch caller ===\" && grep -rn \"execute_dummy_batch\" patch/platform/patch_engine_core.py | head",
  "description": "Check pd channel, edge cloud flag, dummy caller"
}
```

</details>


---

## User <sup>2026/8/5 20:57:48</sup>

<details>
<summary>Tool Result</summary>

```
=== _pp_pd_channel assignment (current) ===
131:    self._pp_pd_channel = None
190:        self._pp_pd_channel = PPSchedulerZmqChannel(
1274:        self._pp_pd_channel = None
=== enable_edge_cloud flag ===
c:/theroadtofree_github/vllm/vllm/config/parallel.py:194:    enable_edge_cloud: bool = False
c:/theroadtofree_github/vllm/vllm/config/parallel.py:698:        if self.enable_edge_cloud:
c:/theroadtofree_github/vllm/vllm/config/parallel.py:778:        if self.enable_edge_cloud:
c:/theroadtofree_github/vllm/vllm/config/parallel.py:782:                    "when enable_edge_cloud is True."
c:/theroadtofree_github/vllm/vllm/distributed/device_communicators/shm_broadcast.py:837:            and vllm_config.parallel_config.enable_edge_cloud
=== is_edge_device def in parallel_state ===
670:    if parallel_config.enable_edge_cloud:
825:    if parallel_config.enable_edge_cloud:
=== execute_dummy_batch caller ===
352:    ``_patched_execute_dummy_batch`` / ``_publish_pd_dummy_zmq``) bypass
1390:#
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:58:08</sup>


---

## Assistant <sup>2026/8/5 20:58:10</sup>

**重大发现：** `PD_MIX = "pd_mix"` 是**默认**的 batch 类型（混合 prefill+decode，output.py 的第 169/401 行）——这是一个具有实际 token 的真实 batch。但它**既不在** `_BT_COORD_ID` map 中，也**不在** `_dp_batch_type_id` 的 `_bt_map` 中（这两个 map 只包含 EMPTY/PF/PL/DF/DL）。因此，**pd_mix 在两次 all_reduce 中都被映射为 0 (EMPTY)** —— dp0 在执行实际工作时报告 `my_bt=0`。

此外，`enable_edge_cloud` 默认为 False。我来检查一下 `_pp_pd_channel` 的设置门控，并阅读 `execute_dummy_batch` 的路径，以找出是什么在向 dp1 输入 dummy 数据。


---

## Assistant <sup>2026/8/5 20:58:11</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py",
  "limit": 90,
  "offset": 120
}
```

</details>


---

## User <sup>2026/8/5 20:58:11</sup>

<details>
<summary>Tool Result</summary>

```
120	        logger.info(
121	            "Edge-cloud mode enabled (pd_separation=%s)",
122	            pd_enabled,
123	        )
124	
125	    # Load PD-separation configuration from environment variables
126	    from vllm_ascend.pd_separation_config import PDSeparationConfig
127	    pd_config = PDSeparationConfig.from_env()
128	
129	    self.step_cnt = 0
130	    # Edge-cloud PD-separation bidirectional ZMQ channel (edge side).
131	    self._pp_pd_channel = None
132	    if pd_enabled and getattr(parallel_config, "is_edge_node", False):
133	        dp_rank = getattr(parallel_config, "data_parallel_rank", 0)
134	        import torch.distributed as dist
135	        from datetime import timedelta
136	
137	        # Cloud cross-DP coord group: edge DP0 hosts a tiny IP-exchange
138	        # store so cloud DP1+ can discover cloud DP0's reachable IP (cloud
139	        # DP0 is the gloo rank-0 / store master). Hosted with
140	        # wait_for_workers=False so the constructor returns immediate
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:58:11</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py",
  "limit": 70,
  "offset": 1385
}
```

</details>


---

## User <sup>2026/8/5 20:58:11</sup>

<details>
<summary>Tool Result</summary>

```
1385	        req = self.input_queue.get_nowait()
1386	        self._handle_client_request(*req)
1387	
1388	
1389	# =======================================================================#
1390	# EngineCore.execute_dummy_batch - 方案③: route dummy per-DP via zmq.       #
1391	# =======================================================================#
1392	def _patched_execute_dummy_batch(self):
1393	    """PD-separation edge: mirror execute_model's per-DP zmq path so the
1394	    idle DP's dummy does NOT reach the cloud via the cross-node
1395	    rpc_broadcast_mq broadcast (which would deliver it to the DP running
1396	    real work and break cross-DP all_reduce pairing -> deadlock).
1397	
1398	    Cloud workers skip the cross-node ``execute_dummy_batch`` (see
1399	    multiproc_executor.worker_busy_loop), so ``executor.execute_dummy_batch``
1400	    below only runs the dummy on *this* DP's edge workers. We additionally
1401	    publish a dummy SchedulerOutput via zmq so the paired cloud 
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:58:13</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github/vllm-ascend/vllm_ascend\" && echo \"=== caller of execute_dummy_batch / when dummy fires in busy loop ===\" && grep -n \"execute_dummy_batch\\|is_ec_consumer\\|_is_coordinated_dp\\|enable_edge_cloud\\|edge_cloud\" patch/platform/patch_engine_core.py | head -50",
  "description": "Find dummy dispatch caller and edge-cloud gating"
}
```

</details>


---

## User <sup>2026/8/5 20:58:14</sup>

<details>
<summary>Tool Result</summary>

```
=== caller of execute_dummy_batch / when dummy fires in busy loop ===
105:    # ``additional_config.edge_cloud_config.pd_separation``. ``init_ascend_config``
111:    edge_cloud = getattr(ascend_config, "edge_cloud_config", None)
113:        edge_cloud is not None
114:        and getattr(edge_cloud, "enabled", False)
115:        and getattr(edge_cloud, "pd_separation", None) is not None
116:        and edge_cloud.pd_separation.enabled
119:    if getattr(parallel_config, "enable_edge_cloud", False):
352:    ``_patched_execute_dummy_batch`` / ``_publish_pd_dummy_zmq``) bypass
428:    self._clear_pending_edge_cloud_draft_for_finished_requests()
448:def _advance_edge_cloud_draft(
477:        state = getattr(model_output, "edge_cloud_draft_state", None)
545:        model_output, "edge_cloud_draft_token_ids", None
551:def _clear_pending_edge_cloud_draft_for_finished_requests(self) -> None:
591:        "clear_pending_edge_cloud_draft_for_req_ids",
598:def _register_edge_cloud_draft_parent(
606
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:59:43</sup>


---

## Assistant <sup>2026/8/5 20:59:47</sup>

关键门控事实：**`_pp_pd_channel` 仅在 `pd_enabled and is_edge_node` 时设置**（第 132 行）—— 即仅在 *edge* 节点上。因此 `_is_coordinated_dp()`（需要 `_pp_pd_channel`）仅在 edge 节点为 True。这意味着：
- 如果确实是非边云场景（`is_edge_node=False` 对两者皆成立）→ `_pp_pd_channel=None` → 协调关闭。
- 如果是边云场景（dp0=edge, dp1=cloud）→ `_pp_pd_channel` 仅在 dp0 上设置 → **协调仅在 dp0 上开启，但协调 `all_reduce` 横跨 dp_group（两个 DP）→ dp1 从不调用它 → dp0 的协调 `all_reduce` 挂起**。

但 dp0 的栈显示它在 *forward*（reduce_scatter）中，位于协调之后，所以协调已完成。挂起发生在别处。我需要确定地确认是否开启了边云模式，并找到 `is_edge_device`。让我搜索日志和代码以寻找明确信号。


---

## Assistant <sup>2026/8/5 20:59:49</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== full log line count ===\" && wc -l log/0805-1.txt && echo \"=== config / edge-cloud signals in log ===\" && grep -in \"edge-cloud mode enabled\\|enable_edge_cloud\\|is_edge_node\\|pd_separation=\\|Edge-cloud\\|edge_cloud_config\\|role.*edge\\|role.*cloud\\|coordinated\\|_coord_winner\\|COORD-IN\\|COORD\\]\" log/0805-1.txt | head -40 && echo \"=== is_edge_device def (vllm tree too) ===\" && grep -rn \"def is_edge_device\\|def is_edge_cloud_pp_mode\\|def is_edge_node\" vllm-ascend/vllm_ascend/ vllm/vllm/ 2>/dev/null | head",
  "description": "Search log for config signals and find is_edge_device def"
}
```

</details>


---

## User <sup>2026/8/5 20:59:51</sup>

<details>
<summary>Tool Result</summary>

```
=== full log line count ===
265 log/0805-1.txt
=== config / edge-cloud signals in log ===
170:(Worker_DP1_TP5_EP13 pid=683679) INFO 08-05 12:28:50 [model_runner_v1.py:7690] [DPDBG] _dummy_run: dp_rank=1 role=cloud peer_bt=0 skip_head=True skip_tail=True
173:(Worker_DP1_TP4_EP12 pid=683636) INFO 08-05 12:28:50 [model_runner_v1.py:7690] [DPDBG] _dummy_run: dp_rank=1 role=cloud peer_bt=0 skip_head=True skip_tail=True
176:(Worker_DP1_TP6_EP14 pid=683722) INFO 08-05 12:28:50 [model_runner_v1.py:7690] [DPDBG] _dummy_run: dp_rank=1 role=cloud peer_bt=0 skip_head=True skip_tail=True
179:(Worker_DP1_TP0_EP8 pid=683488) INFO 08-05 12:28:50 [model_runner_v1.py:7690] [DPDBG] _dummy_run: dp_rank=1 role=cloud peer_bt=0 skip_head=True skip_tail=True
195:(Worker_DP1_TP1_EP9 pid=683507) INFO 08-05 12:28:50 [model_runner_v1.py:7690] [DPDBG] _dummy_run: dp_rank=1 role=cloud peer_bt=0 skip_head=True skip_tail=True
201:(Worker_DP1_TP7_EP15 pid=683765) INFO 08-05 12:28:50 [model_runner_v1.py:7690] [DPDBG] _
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 20:59:51</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github/vllm-ascend/vllm_ascend\" && echo \"=== is_edge_device import source in model_runner ===\" && sed -n '40,60p' worker/model_runner_v1.py && echo \"=== grep is_edge_device def in whole vllm-ascend incl subdirs ===\" && grep -rn \"def is_edge_device\" . 2>/dev/null; grep -rn \"^def is_edge_device\\|^    def is_edge_device\\|is_edge_device =\" distributed/ parallel_state.py 2>/dev/null | head",
  "description": "Find is_edge_device import and definition"
}
```

</details>


---

## User <sup>2026/8/5 20:59:53</sup>

<details>
<summary>Tool Result</summary>

```
=== is_edge_device import source in model_runner ===
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_edge_cloud_layer_range,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    is_edge_cloud_pp_mode,
    is_edge_device,
    set_edge_cloud_layer_range,
)
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
=== grep is_edge_device def in whole vllm-ascend incl subdirs ===
```

</details>


---

## Assistant <sup>2026/8/5 21:01:08</sup>


---

## Assistant <sup>2026/8/5 21:01:11</sup>

从日志中确认的关键点：**没有 `[COORD-IN]`/`_coord_winner` 日志行** —— 说明协调机制处于关闭状态（非边云模式，`_pp_pd_channel=None`）。挂死发生在 **dummy 路径** (`_dummy_run` → `_sync_metadata_across_dp`) 与 **实际前向计算** (MLA reduce_scatter) 的交互中。同时确认：`is_edge_device` 位于 [vllm/vllm/distributed/parallel_state.py:1249](vllm/vllm/distributed/parallel_state.py#L1249)。

关键点在于：EP=16 跨越了两台服务器（EP0-7 在 dp0 上，EP8-15 在 dp1 上），因此 **MoE EP all-toall 是跨 DP 的**。如果 dp1 的 dummy 跳过了前向计算，而 dp0 的实际计算需要 MoE all-toall，则会导致挂死。让我验证一下：读取 `is_edge_device`，检查 reduce_scatter 组，并将 `_dummy_run` 的跳过逻辑与参考代码进行对比。


---

## Assistant <sup>2026/8/5 21:01:12</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm/vllm/distributed/parallel_state.py",
  "limit": 30,
  "offset": 1245
}
```

</details>


---

## User <sup>2026/8/5 21:01:12</sup>

<details>
<summary>Tool Result</summary>

```
1245	_EDGE_CLOUD_HEAD_K: int = 0
1246	_EDGE_CLOUD_TAIL_K: int = 0
1247	
1248	
1249	def is_edge_device() -> bool:
1250	    return _IS_EDGE_DEVICE is True
1251	
1252	
1253	def is_cloud_device() -> bool:
1254	    return _IS_EDGE_DEVICE is False
1255	
1256	
1257	def is_edge_cloud_pp_mode() -> bool:
1258	    return _IS_EDGE_DEVICE is not None
1259	
1260	
1261	def is_edge_cloud_first_stage(intermediate_tensors) -> bool:
1262	    return is_edge_cloud_pp_mode() and intermediate_tensors is None
1263	
1264	
1265	def set_edge_cloud_layer_range(head_k: int, tail_k: int) -> None:
1266	    """Store the edge-cloud head / tail layer counts for use by
1267	    :func:`make_layers` during model initialisation.
1268	
1269	    Must be called **before** ``initialize_model()`` / ``get_model()``
1270	    so that ``make_layers()`` can create the correct mix of real layers
1271	    and ``PPMissingLayer`` placeholders directly on the target device.
1272	    """
1273	    global _EDGE_CLOUD_HEAD_K, _EDGE_CLOUD_TAI
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:01:15</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && CUR=\"vllm-ascend/vllm_ascend/worker/model_runner_v1.py\" && REF=\"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/worker/model_runner_v1.py\" && echo \"=== diff of _dummy_run skip-both logic region (current vs ref) ===\" && diff <(tr -d '\\r' < \"$CUR\") <(tr -d '\\r' < \"$REF\") > /tmp/mr_diff.txt 2>&1; wc -l /tmp/mr_diff.txt; echo \"=== lines mentioning skip_head/peer_bt/dummy/sync_meta in diff ===\" && grep -n \"skip_head\\|skip_tail\\|_peer_bt\\|peer_bt\\|_dummy_run\\|sync_metadata\\|_dp_batch_type_id\\|peer_batch_type\\|both DUMMY\\|skip both\\|all-toall\\|all_toall\" /tmp/mr_diff.txt | head -60",
  "description": "Diff model_runner vs reference, find dummy/sync changes"
}
```

</details>


---

## User <sup>2026/8/5 21:01:19</sup>

<details>
<summary>Tool Result</summary>

```
=== diff of _dummy_run skip-both logic region (current vs ref) ===
548 /tmp/mr_diff.txt
=== lines mentioning skip_head/peer_bt/dummy/sync_meta in diff ===
3:<         # 2=PL, 3=DF, 4=DL). Packed into sync_metadata all_reduce so the
5:<         self._dp_batch_type_id = 0
6:<         self._peer_batch_type_id = 0
31:<         packed_tensor[2][self.dp_rank] = self._dp_batch_type_id
34:<         self._peer_batch_type_id = int(packed_tensor[2, 1 - self.dp_rank].item())
36:<         # [DPDBG] trace sync_metadata pairing to locate cross-DP misalignment.
39:<         # means this side is a DUMMY, 1/2/3/4 = real PF/PL/DF/DL. peer_bt is
48:<             "[PP-EVT] sync_meta: rank=%s dp_rank=%s call=%s my_bt=%s peer_bt=%s "
50:<             _sm_rank, self.dp_rank, _smc, self._dp_batch_type_id,
51:<             self._peer_batch_type_id, num_tokens,
56:<         # 1=PF, 2=PL, 3=DF, 4=DL). Packed into sync_metadata all_reduce
63:<         self._dp_batch_type_id = _bt_map.get(scheduler_output.batch_typ
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:02:39</sup>


---

## Assistant <sup>2026/8/5 21:02:42</sup>

**这就是回归问题所在。** `model_runner_v1.py` 的差异显示，整个 `_dp_batch_type_id`/`_peer_batch_type_id` 交换以及 `_dummy_run` 中的 `skip-both` 逻辑在当前代码中是**新增的**（参考代码中没有）。关键在于，当前代码 211-218 行的注释明确描述了该失败模式：

> *"peer_bt=0, treats this as DUMMY and skips the all-toall, while this cloud runs the real all-toall -> cross-DP all-toall mismatch"*

该修复方案仅涵盖了 `_bt_map` 中的 batch types。**`PD_MIX`（非边云默认值）不在该 map 中**，因此实际的 `PD_MIX` 批次报告的 `_dp_batch_type_id=0` —— 这与 idle/dummy 无法区分。对等节点的 dummy 因此看到了 `peer_bt=0` 并**跳过了前向计算**，破坏了跨 DP 集合通信的配对。

我来确认一下确切的集合通信机制，并验证参考代码中 dummy 的行为（全前向计算）。


---

## Assistant <sup>2026/8/5 21:02:43</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm/vllm/distributed/communication_op.py",
  "limit": 45,
  "offset": 1
}
```

</details>


---

## User <sup>2026/8/5 21:02:43</sup>

<details>
<summary>Tool Result</summary>

```
1	# SPDX-License-Identifier: Apache-2.0
2	# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
3	
4	from typing import Any
5	
6	import torch
7	import torch.distributed
8	
9	from .parallel_state import get_tp_group
10	
11	
12	def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
13	    """All-reduce the input tensor across model parallel group."""
14	    return get_tp_group().all_reduce(input_)
15	
16	
17	def tensor_model_parallel_all_gather(
18	    input_: torch.Tensor, dim: int = -1
19	) -> torch.Tensor:
20	    """All-gather the input tensor across model parallel group."""
21	    return get_tp_group().all_gather(input_, dim)
22	
23	
24	def tensor_model_parallel_reduce_scatter(
25	    input_: torch.Tensor, dim: int = -1
26	) -> torch.Tensor:
27	    """Reduce-Scatter the input tensor across model parallel group."""
28	    return get_tp_group().reduce_scatter(input_, dim)
29	
30	
31	def tensor_model_parallel_gather(
32	    input_: torch.Tensor, dst: 
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:02:43</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/ops/linear_op.py",
  "limit": 45,
  "offset": 545
}
```

</details>


---

## User <sup>2026/8/5 21:02:43</sup>

<details>
<summary>Tool Result</summary>

```
545	            isinstance(self.layer.quant_method, AscendLinearMethod)
546	            and isinstance(self.layer.quant_method.quant_method, AscendW8A8LinearMethod)
547	        ):
548	            if x.dtype != torch.int8:
549	                x_quant = torch.ops.vllm.quantize(
550	                    x,
551	                    self.layer.aclnn_input_scale,
552	                    self.layer.aclnn_input_scale_reciprocal,
553	                    self.layer.aclnn_input_offset,
554	                )
555	            else:
556	                x_quant = x
557	            quant_bias = self.layer.quant_bias
558	            deq_scale = self.layer.deq_scale
559	            output_dtype = torch.bfloat16
560	            output = torch_npu.npu_mm_reduce_scatter_base(
561	                x_quant,
562	                self.layer.weight,
563	                hcom_name,
564	                world_size,
565	                reduce_op="sum",
566	                bias=None,
567	                comm_turn=0,
568	 
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:02:45</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && REF=\"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/worker/model_runner_v1.py\" && echo \"=== reference _dummy_run: does it skip forward or always run? ===\" && grep -n \"def _dummy_run\\|skip_head\\|skip_tail\\|_peer_bt\\|peer_bt\\|_skip_head\\|both DUMMY\\|_model_forward\\|hidden_states = None\" \"$REF\" | head -30 && echo \"=== reference _sync_metadata_across_dp bt exchange? ===\" && grep -n \"_dp_batch_type_id\\|_peer_batch_type_id\\|batch_type_id\\|packed_tensor\\[2\\]\" \"$REF\" | head",
  "description": "Check reference dummy behavior and reduce_scatter group"
}
```

</details>


---

## User <sup>2026/8/5 21:03:47</sup>

<details>
<summary>Tool Result</summary>

```
=== reference _dummy_run: does it skip forward or always run? ===
4160:            hidden_states = self._model_forward(
4166:            aux_hidden_states = None
4175:                    aux_hidden_states = None
4182:                    aux_hidden_states = None
5626:    def _model_forward(
5836:        going directly to _preprocess (sync_and_gather) + _model_forward.
6243:            hidden_states = self._model_forward(
6531:        self._eagle3_cloud_aux_hidden_states = None
7123:    def _dummy_run(
7416:                outputs = self._model_forward(
7490:                    outputs = self._model_forward(
=== reference _sync_metadata_across_dp bt exchange? ===
```

</details>


---

## Assistant <sup>2026/8/5 21:06:12</sup>


---

## Assistant <sup>2026/8/5 21:06:14</sup>

确认：参考 `_dummy_run` 执行的是**完整前向传播**（参考代码第 7416/7490 行调用了 `_model_forward`，没有跳过逻辑），且参考的 `_sync_metadata_across_dp` **没有 bt_id 交换**。跳过逻辑和 bt_id 交换是新增的修改。此外，`tensor_model_parallel_reduce_scatter` 使用了 `get_tp_group()`（8 个 rank，服务器内部）。

现在，关于 dp0 的 TP `reduce_scatter` 挂起，还有一个谜团。我来检查一下 `matmul_and_reduce` 的 comm group/`world_size`（它是否可能是跨 DP 的，例如 CP/EP 组？），并验证 `_IS_EDGE_DEVICE` 的门控机制，以便设计修复方案。


---

## Assistant <sup>2026/8/5 21:06:15</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/ops/linear_op.py",
  "limit": 75,
  "offset": 470
}
```

</details>


---

## User <sup>2026/8/5 21:06:15</sup>

<details>
<summary>Tool Result</summary>

```
470	        else:
471	            output = output_parallel
472	        output_bias = self.bias if self.skip_bias_add else None
473	        return output, output_bias
474	
475	
476	class SequenceRowParallelOp(CustomRowParallelOp):
477	    def __init__(self, layer):
478	        super().__init__(layer)
479	        self.unique_prefix = None
480	
481	    def apply_impl(self, input_: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
482	        """Linear layer with column parallelism.
483	
484	        Implemented multiple optimization projects for dense models, such as FlashComm and
485	        communication-computation fusion.
486	        """
487	        input_parallel = self.get_input_parallel(input_)
488	
489	        assert self.quant_method is not None
490	        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
491	
492	        if self.tp_size == 1 or not self.reduce_results:
493	            output = self.quant_method.apply(self.layer, input_p
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:06:17</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== _IS_EDGE_DEVICE set sites ===\" && grep -rn \"_IS_EDGE_DEVICE\\s*=\\|_IS_EDGE_DEVICE=\" vllm/vllm/distributed/parallel_state.py | head && echo \"=== context parallel / CP group config (cross-DP?) ===\" && grep -rn \"context_parallel\\|cp_size\\|get_cp_group\\|enable_cp\\|cp_world\\|ShardedCP\" vllm/vllm/config/parallel.py vllm-ascend/vllm_ascend/ops/linear_op.py 2>/dev/null | head -20 && echo \"=== how matmul_and_reduce gets world_size/hcom (the class) ===\" && grep -n \"class .*RowParallelOp\\|class .*ColParallelOp\\|def matmul_and_reduce\\|world_size\\|hcom_name\\|comm_group\\|def apply\" vllm-ascend/vllm_ascend/ops/linear_op.py | head -40",
  "description": "Check _IS_EDGE_DEVICE gating and reduce_scatter group"
}
```

</details>


---

## User <sup>2026/8/5 21:06:19</sup>

<details>
<summary>Tool Result</summary>

```
=== _IS_EDGE_DEVICE set sites ===
1689:        _IS_EDGE_DEVICE = (rank == 0)
1814:        _IS_EDGE_DEVICE = is_edge
2252:    _IS_EDGE_DEVICE = None
=== context parallel / CP group config (cross-DP?) ===
vllm/vllm/config/parallel.py:115:    prefill_context_parallel_size: int = 1
vllm/vllm/config/parallel.py:343:    decode_context_parallel_size: int = 1
vllm/vllm/config/parallel.py:346:    needs to be divisible by dcp_size."""
vllm/vllm/config/parallel.py:365:    For `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`,
vllm/vllm/config/parallel.py:366:        and `total_cp_world_size = pcp_world_size * dcp_world_size`.
vllm/vllm/config/parallel.py:370:        total_cp_rank `i % total_cp_world_size`.
vllm/vllm/config/parallel.py:503:        # parallel(DCP), tp_size needs to be divisible by dcp_size,
vllm/vllm/config/parallel.py:506:        # tp_size//dcp_size DCP groups.
vllm/vllm/config/parallel.py:507:        if self.tensor_parallel_size % self.decode_context_parallel_size != 0:
vllm
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:08:29</sup>


---

## Assistant <sup>2026/8/5 21:08:31</sup>

确认：MLA `o_proj` 的 `reduce_scatter` 使用了 `get_tp_group()`，且 `world_size = self.layer.tp_size` (8) —— 这是服务器内部的 TP，并非跨 DP。因此，dp0 的 `reduce_scatter` 挂起意味着 dp0 的 8 个 rank 出现了分歧。参考的 `_dummy_run` 执行了**完整前向传播**（在 7416/7490 行调用了 `_model_forward`，没有跳过）—— 与真实前向传播配对。当前的跳过两者逻辑移除了这一点。

让我阅读一下 `_determine_batch_execution_and_padding`（两个路径都会调用的函数）以及当前的 `_dummy_run` 头部，以查明确切的集合通信配对，并确认跨 DP 的 EP 通信机制。


---

## Assistant <sup>2026/8/5 21:08:32</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
  "limit": 120,
  "offset": 6843
}
```

</details>


---

## User <sup>2026/8/5 21:08:32</sup>

<details>
<summary>Tool Result</summary>

```
6843	    def _determine_batch_execution_and_padding(
6844	        self,
6845	        num_tokens: int,
6846	        num_reqs: int,
6847	        num_scheduled_tokens_np: np.ndarray,
6848	        max_num_scheduled_tokens: int,
6849	        use_cascade_attn: bool,
6850	        allow_microbatching: bool = False,
6851	        force_eager: bool = False,
6852	        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
6853	        # be improved in model runner v2)
6854	        force_uniform_decode: bool | None = None,
6855	        force_has_lora: bool | None = None,
6856	        force_num_active_loras: int | None = None,
6857	        num_encoder_reqs: int = 0,
6858	        for_cudagraph_capture: bool = False,
6859	    ) -> tuple[CUDAGraphMode, BatchDescriptor, bool, torch.Tensor | None, CUDAGraphStat | None]:
6860	        num_tokens_padded = self._pad_for_sequence_parallelism(
6861	            num_tokens, for_cudagraph_capture=for_cudagraph_capture
6862	        )
6863
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:10:14</sup>


---

## Assistant <sup>2026/8/5 21:10:15</sup>

`_determine_batch_execution_and_padding` 在 DP>1 时（第 6906-6907 行）会调用一次 `_sync_metadata_across_dp`——这是真实路径和 dummy 路径在向前传播（forward）之前调用的**唯一**集合通信。在此之后，dummy 会跳过向前传播，而真实路径则会执行（包括跨 DP 的 MoE EP all-toall）。我来确认一下 dummy 的流程以及 EP all-toall 是否为跨 DP 的，并验证 `_dummy_run` 的头部。


---

## Assistant <sup>2026/8/5 21:10:16</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
  "limit": 90,
  "offset": 7332
}
```

</details>


---

## User <sup>2026/8/5 21:10:16</sup>

<details>
<summary>Tool Result</summary>

```
7332	    def _dummy_run(
7333	        self,
7334	        num_tokens: int,
7335	        with_prefill: bool = False,
7336	        cudagraph_runtime_mode: CUDAGraphMode | None = None,
7337	        force_attention: bool = False,
7338	        uniform_decode: bool = False,
7339	        is_profile: bool = False,
7340	        create_mixed_batch: bool = False,
7341	        allow_microbatching: bool = True,
7342	        skip_eplb: bool = False,
7343	        remove_lora: bool = True,
7344	        is_graph_capturing: bool = False,
7345	        num_active_loras: int = 0,
7346	        profile_seq_lens: int | None = None,
7347	        profile_cpp: bool = False,
7348	        layer_slice_info: Any = None,
7349	    ) -> tuple[torch.Tensor, torch.Tensor]:
7350	        # only support eager mode and piecewise graph now
7351	        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()
7352	
7353	        # [SLICE-DIAG] Log whether _dummy_run receives layer_slice_info.
7354	  
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:10:20</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== EP group: cross-DP? (expert parallel all-toall group composition) ===\" && grep -rn \"get_expert_parallel_group\\|ep_group\\|expert_parallel.*world\\|all_to_all_single\\|def.*ep.*group\\|EPGroup\\|get_ep_group\" vllm/vllm/distributed/parallel_state.py 2>/dev/null | head -15 && echo \"=== MoE all-toall call sites in vllm-ascend ===\" && grep -rn \"all_to_all_single\\|all_toall\\|a2a\\|all_to_all\" vllm-ascend/vllm_ascend/ops/ vllm-ascend/vllm_ascend/model_executor/layers/fused_moe.py 2>/dev/null | head -15 && echo \"=== is edge-cloud / PD-sep on? check pd_mix default + coord off already confirmed ===\" && echo \"=== reference _dummy_run: does it call _model_forward unconditionally? (lines 7123-7160) ===\" && sed -n '7380,7420p' \"C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/worker/model_runner_v1.py\"",
  "description": "Check EP all-toall cross-DP and reference dummy flow"
}
```

</details>


---

## User <sup>2026/8/5 21:10:22</sup>

<details>
<summary>Tool Result</summary>

```
=== EP group: cross-DP? (expert parallel all-toall group composition) ===
1219:def _replace_active_groups(
1322:def get_ep_group() -> GroupCoordinator:
1334:def get_eplb_group() -> GroupCoordinator:
=== MoE all-toall call sites in vllm-ascend ===
vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py:26:def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py:29:        a2a_out = torch.empty_like(input_)
vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py:32:        a2a_out = input_.new_empty(
vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py:45:            handle = dist.all_to_all_single(
vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py:46:                a2a_out,
vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py:54:        handle = dist.all_to_all_single(
vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py:55:            a2a_out,
vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py:62:    return inp
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:15:47</sup>


---

## Assistant <sup>2026/8/5 21:15:48</sup>

现在我已经掌握了完整且确凿的情况。让我核实最后几个细节：确认 `is_edge_cloud_pp_mode()` 在非边云场景下为 False，并检查 `_dummy_run` 的末尾是否存在 forward 后的集合通信（collectives）。


---

## Assistant <sup>2026/8/5 21:15:50</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== _IS_EDGE_DEVICE init: None unless edge-cloud init ===\" && sed -n '2248,2253p;1685,1692p;1810,1816p' vllm/vllm/distributed/parallel_state.py && echo \"=== is_edge_cloud_pp_mode imported in model_runner? ===\" && grep -n \"is_edge_cloud_pp_mode\" vllm-ascend/vllm_ascend/worker/model_runner_v1.py | head -3 && echo \"=== _dummy_run tail: any post-forward collective? (7820-7920) ===\" && sed -n '7820,7920p' vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
  "description": "Verify edge-cloud gating and dummy tail collectives"
}
```

</details>


---

## User <sup>2026/8/5 21:15:51</sup>

<details>
<summary>Tool Result</summary>

```
=== _IS_EDGE_DEVICE init: None unless edge-cloud init ===
            f"(= 1 + data_parallel_size={data_parallel_size} * "
            f"cloud_npu_count={cloud_npu_count}).")

        # The single edge rank is always rank 0 in this mode.
        _IS_EDGE_DEVICE = (rank == 0)
        is_edge = _IS_EDGE_DEVICE

        # TP groups:
        world_size_per_instance = edge_npu_count + cloud_npu_count
        # Use local_rank within dp instance to determine is_edge
        local_rank = rank % world_size_per_instance
        is_edge = local_rank < edge_npu_count
        _IS_EDGE_DEVICE = is_edge

        # Build TP group: each side forms its own TP group within each dp instance

    if _TP:
        _TP.destroy()
    _TP = None
    _IS_EDGE_DEVICE = None

=== is_edge_cloud_pp_mode imported in model_runner? ===
51:    is_edge_cloud_pp_mode,
4362:            if is_edge_cloud_pp_mode() and isinstance(hidden_states, IntermediateTensors) and not is_edge_device():
=== _dummy_run tail: any post-forwa
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:16:12</sup>


---

## Assistant <sup>2026/8/5 21:16:14</sup>

已确认所有关键事实：
- `_IS_EDGE_DEVICE = None` 默认情况下（第 2252 行）-> `is_edge_cloud_pp_mode()` 在非边云场景下为 **False**。仅在边云初始化路径（1689/1814）中设置。
- `is_edge_cloud_pp_mode` 已在 model_runner 中导入（第 51 行）。
- 当 skip 被禁用时（`_skip_head=False`），head 代码块会以 `layer_slice_info=None` 运行 `_model_forward` -> **完整模型前向传播**，与参考（baseline）逻辑一致。

现在我已经有了明确的根本原因和修复方案。让我来应用这个修复。


---

## Assistant <sup>2026/8/5 21:16:17</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
  "old_string": "            hidden_states = None\n            _skip_head = False\n            _skip_tail = False\n            if not is_profile and not is_graph_capturing:\n                _peer_bt = self._peer_batch_type_id",
  "new_string": "            hidden_states = None\n            _skip_head = False\n            _skip_tail = False\n            # [非边云 fix] 段级 skip 仅适用于边云 PD-separation（head/middle/tail 分段）。\n            # 非边云 DP>1（base scheduler，batch_type=PD_MIX）下必须跑完整 forward：\n            # PD_MIX 不在 _bt_map（仅 EMPTY/PF/PL/DF/DL），_dp_batch_type_id 落到 0，与\n            # DUMMY/idle 不可区分；若放行 skip-both，idle 侧 dummy 见 peer_bt==0 跳过整个\n            # forward，不再配对 real 侧的跨 DP EP all-toall 等集合通信 -> 集合通信计数错位死锁\n            # （单请求即挂：dp1 dummy 卡在 sync_metadata all_reduce，dp0 real 卡在 forward）。\n            # 故非边云一律 _skip_head=_skip_tail=False，走完整 _model_forward（与 baseline 一致）。\n            if (\n                not is_profile\n                and not is_graph_capturing\n                and is_edge_cloud_pp_mode()\n            ):\n                _peer_bt = self._peer_batch_type_id"
}
```

</details>


---

## User <sup>2026/8/5 21:16:18</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/model_runner_v1.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 21:16:39</sup>


---

## Assistant <sup>2026/8/5 21:16:40</sup>

我来核实一下修复方案，并检查是否还有其他依赖于 `bt_id` 的 skip sites 可能导致非边云场景出现问题。


---

## Assistant <sup>2026/8/5 21:16:42</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github/vllm-ascend/vllm_ascend\" && echo \"=== all skip_head/skip_tail sites (confirm only one block) ===\" && grep -n \"_skip_head\\|_skip_tail\\|skip_head\\|skip_tail\" worker/model_runner_v1.py && echo \"=== verify the edited region ===\" && sed -n '7671,7700p' worker/model_runner_v1.py && echo \"=== worker.execute_dummy_batch dispatch (1721-1740) ===\" && sed -n '1715,1740p' worker/worker.py",
  "description": "Verify fix and check for other skip sites"
}
```

</details>


---

## User <sup>2026/8/5 21:16:43</sup>

<details>
<summary>Tool Result</summary>

```
=== all skip_head/skip_tail sites (confirm only one block) ===
7672:            _skip_head = False
7673:            _skip_tail = False
7680:            # 故非边云一律 _skip_head=_skip_tail=False，走完整 _model_forward（与 baseline 一致）。
7691:                    _skip_head = True
7692:                    _skip_tail = True
7695:                        _skip_head = True
7697:                        _skip_tail = True
7703:                    "skip_head=%s skip_tail=%s",
7706:                    _peer_bt, _skip_head, _skip_tail,
7709:            if not _skip_head:
7814:            if is_edge_device() and not _skip_tail:
=== verify the edited region ===
            hidden_states = None
            _skip_head = False
            _skip_tail = False
            # [非边云 fix] 段级 skip 仅适用于边云 PD-separation（head/middle/tail 分段）。
            # 非边云 DP>1（base scheduler，batch_type=PD_MIX）下必须跑完整 forward：
            # PD_MIX 不在 _bt_map（仅 EMPTY/PF/PL/DF/DL），_dp_batch_type_id 落到 0，与
            # DUMMY/idle 不可区分；若放行 ski
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 21:17:11</sup>


---

## Assistant <sup>2026/8/5 21:17:12</sup>

修复已验证：只有一个 `skip-both` 代码块（现在已受限），并且 `worker.execute_dummy_batch` 调用 `_dummy_run` 时传入了 `layer_slice_info=None`（非边云）-> 在应用修复后，head 代码块会运行完整的 `_model_forward`（参考行为）。

我来把这个发现记录在内存中，然后展示完整的诊断结果。


---

## Assistant <sup>2026/8/5 21:17:23</sup>

<details>
<summary>Tool: <code>Write</code></summary>

```json
{
  "file_path": "C:\\Users\\root\\.claude\\projects\\c--theroadtofree-github\\memory\\dp-dummy-skip-nonedgecloud-hang.md",
  "content": "---\nname: dp-dummy-skip-nonedgecloud-hang\ndescription: 非边云 DP>1 单请求挂死根因 - _dummy_run skip-both 误跳 forward 致跨 DP 集合通信错位\nmetadata:\n  type: project\n---\n\n非边云 DP=2（两服务器各 8 卡，dp0/dp1）+ MoE（kimi25_w4a8_static_m6，EP 跨 DP）+ base scheduler（batch_type=PD_MIX），单请求即挂死。\n\n**根因**：本工程为边云 DP 新增的 `_dummy_run` 段级 skip-both 逻辑 + `_sync_metadata_across_dp` 的 `_dp_batch_type_id` 交换（baseline `C:\\yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy` 均无）在非边云被误触发。\n- `_bt_map` / `_BT_COORD_ID` 只含 EMPTY/PF/PL/DF/DL，**PD_MIX 不在内** -> `_bt_map.get(PD_MIX,0)=0`。real PD_MIX batch 报 `_dp_batch_type_id=0`，与 DUMMY/idle 不可区分。\n- idle 侧 dummy 在 `_sync_metadata_across_dp` 见 `peer_bt==0` -> `_skip_head=_skip_tail=True`，**跳过整个 forward**（含跨 DP EP all-toall），只留 sync_metadata。\n- real 侧 forward 的跨 DP 集合通信失去 dummy 配对 -> 计数错位死锁：dp1 dummy 卡在 sync_metadata all_reduce，dp0 real 卡在 forward（栈见 MLA o_proj 的 TP reduce_scatter）。\n\n**关键判定**：非边云下 `enable_edge_cloud=False`/`is_edge_node=False` -> `_pp_pd_channel=None` -> `_is_coordinated_dp()=False`（无 [COORD] 日志，协调未开）；但 `_dummy_run` 的 skip-both **未** gate 边云，故仍触发。日志证据：dp1 `role=cloud peer_bt=0 skip_head=True skip_tail=True`，dp0 `my_bt=0 peer_bt=0`，batch_type=pd_mix。\n\n**修复**（`vllm_ascend/worker/model_runner_v1.py` `_dummy_run`，~7674）：把 skip-both 块的条件 `if not is_profile and not is_graph_capturing:` 改为 `... and is_edge_cloud_pp_mode():`。非边云下 `_IS_EDGE_DEVICE=None` -> `is_edge_cloud_pp_mode()=False`，skip 不触发，`_skip_head/_skip_tail` 恒 False -> head 块以 `layer_slice_info=None` 跑完整 `_model_forward`，与 baseline 一致，配对所有集合通信。边云行为不变（PF/PL/DF/DL 在 map 内，skip 按段配对）。\n\n**未覆盖的隐患**：`_bt_map` 仍缺 PURE_PREFILL/PURE_DECODE（非边云 PDSeparatedScheduler 会用），若开启非边云 PD-separation 仍可能 peer_bt=0 误跳——但当前 skip 已 gate 边云，非边云一律走全 forward，故无死锁风险。关联 [[dp-segment-coordination]]。\n"
}
```

</details>


---

## User <sup>2026/8/5 21:17:24</sup>

<details>
<summary>Tool Result</summary>

```
File created successfully at: C:\Users\root\.claude\projects\c--theroadtofree-github\memory\dp-dummy-skip-nonedgecloud-hang.md
```

</details>


---

## Assistant <sup>2026/8/5 21:17:27</sup>


---

## Assistant <sup>2026/8/5 21:17:27</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "C:\\Users\\root\\.claude\\projects\\c--theroadtofree-github\\memory\\MEMORY.md"
}
```

</details>


---

## User <sup>2026/8/5 21:17:27</sup>

<details>
<summary>Tool Result</summary>

```
<system-reminder>This memory is 2 days old. Memories are point-in-time observations, not live state — claims about code behavior or file:line citations may be outdated. Verify against current code before asserting as fact.</system-reminder>
1	- [DP 段级协调](dp-segment-coordination.md) - 边云 DP prefill 优化：协调粒度 batch_type->segment，消除 one-real-one-dummy
2
```

</details>


---

## Assistant <sup>2026/8/5 21:17:36</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "C:\\Users\\root\\.claude\\projects\\c--theroadtofree-github\\memory\\MEMORY.md",
  "old_string": "- [DP 段级协调](dp-segment-coordination.md) - 边云 DP prefill 优化：协调粒度 batch_type->segment，消除 one-real-one-dummy",
  "new_string": "- [DP 段级协调](dp-segment-coordination.md) - 边云 DP prefill 优化：协调粒度 batch_type->segment，消除 one-real-one-dummy\n- [非边云 DP 挂死](dp-dummy-skip-nonedgecloud-hang.md) - _dummy_run skip-both 未 gate 边云，PD_MIX->bt_id0 误跳 forward 致跨 DP 集合通信错位死锁"
}
```

</details>


---

## User <sup>2026/8/5 21:17:37</sup>

<details>
<summary>Tool Result</summary>

```
The file C:\Users\root\.claude\projects\c--theroadtofree-github\memory\MEMORY.md has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 21:17:40</sup>


---

## Assistant <sup>2026/8/5 21:17:59</sup>

已完成定位与修复。下面是完整诊断。

## 根因

挂死由本工程为**边云 DP** 新增的 `_dummy_run` 段级 **skip-both 逻辑** 在**非边云**场景被误触发导致。对照目录 `C:\yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy`（baseline）确认：这部分逻辑 baseline 没有，是新增的回归。

**两条新增代码（baseline 均无）：**
1. `_sync_metadata_across_dp` 里新增 `_dp_batch_type_id` 跨 DP 交换（[model_runner_v1.py:1692](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1692)）；
2. `_dummy_run` 里新增的 skip-both 段级跳过（[model_runner_v1.py:7674](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L7674)）：`peer_bt==0` 时 `_skip_head=_skip_tail=True`，**跳过整个 forward**。

**致命点：bt_id 映射缺 PD_MIX。** `_bt_map`/`_BT_COORD_ID` 只含 `EMPTY/PF/PL/DF/DL`（[model_runner_v1.py:3559](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L3559)），而本场景 base scheduler 的真实 batch 是 **`PD_MIX`**（[output.py:169](vllm/vllm/v1/core/sched/output.py#L169)），不在表内 → `_bt_map.get(PD_MIX, 0) = 0`。于是 **real PD_MIX batch 也报 `bt_id=0`**，和 DUMMY/idle 完全不可区分。

## 日志证据（与代码完全吻合）

| 侧 | 现象 | 含义 |
|---|---|---|
| dp0（real, pd_mix, 13 tokens） | `sync_meta: my_bt=0 peer_bt=0` | real 却报 bt=0（PD_MIX→0）|
| dp1（dummy） | `[DPDBG] _dummy_run: role=cloud peer_bt=0 skip_head=True skip_tail=True` | 见 peer_bt=0 → 跳过整个 forward |
| dp0 栈 | 卡在 MLA o_proj `tensor_model_parallel_reduce_scatter`（forward 内）| forward 走不下去 |
| dp1 栈 | 卡在 `_sync_metadata_across_dp` all_reduce（`_dummy_run`→`_determine_batch_execution_and_padding`）| 等待 dp0 的下一次 sync |

并且日志里**没有 `[COORD]`/`_coord_winner`**，batch_type 是 `pd_mix`（非 PF/PL/DF/DL）→ 证实这是**非边云**：`enable_edge_cloud=False`/`is_edge_node=False` → `_pp_pd_channel=None` → `_is_coordinated_dp()=False`（协调未开）。但 `_dummy_run` 的 skip-both **没有 gate 边云**，于是照常触发。

## 死锁机制

EP=16 跨两台服务器，MoE EP all-toall 是**跨 DP** 集合通信。每步 real 与 dummy 各做一次 `_sync_metadata_across_dp`（DP all_reduce，1:1 配对），之后：
- **real dp0**：sync → 完整 forward（含跨 DP EP all-toall）；
- **dummy dp1**：sync → **skip 整个 forward**（不参与 EP all-toall），立刻进入下一次 dummy 的 sync。

dummy 瞬间完成、real 的跨 DP all-toall 无人配对 → 集合通信计数错位 → dp1 卡在 sync 等 dp0、dp0 卡在 forward 等 dp1，双向死锁。baseline 的 `_dummy_run` **无条件跑完整 `_model_forward`**（[ref:7416](C:/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L7416)），所以两边集合通信 1:1 配对，不死锁。

## 已应用的修复

[model_runner_v1.py:7674](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L7674) 把 skip-both 块的条件加上边云 gate：

```python
if (not is_profile and not is_graph_capturing and is_edge_cloud_pp_mode()):
    _peer_bt = self._peer_batch_type_id
    if _peer_bt == 0: ...        # skip-both
    elif is_edge_device(): ...   # 段配对
```

- **非边云**：`_IS_EDGE_DEVICE=None` → `is_edge_cloud_pp_mode()=False` → skip 不触发，`_skip_head/_skip_tail` 恒 False → head 块以 `layer_slice_info=None` 跑**完整 `_model_forward`**，与 baseline 一致，配对所有跨 DP 集合通信。
- **边云不变**：PF/PL/DF/DL 都在 `_bt_map` 内，peer_bt 不会误判为 0，skip 按段配对逻辑保留。

`_sync_metadata_across_dp` 的 bt_id 交换在非边云仍保留（仅一点 all_reduce 开销，无功能性影响，且其唯一消费者就是已 gate 的 skip 逻辑）。

## 一个遗留隐患（未动，仅提示）

`_bt_map` 仍缺 `PURE_PREFILL`/`PURE_DECODE`（非边云 PDSeparatedScheduler 会用）。当前因 skip 已整体 gate 边云，非边云一律走全 forward，所以**无死锁风险**；但若将来要在非边云 PD-separation 下复用 bt_id 机制，需补这两个枚举。按 [[dp-segment-coordination]] 里"不要动云侧"的教训，本次只做最小边侧 gate，未碰协调/云侧逻辑。

建议用单请求复跑验证 dp1 不再出现 `skip_head=True`、dp0/dp1 正常出 token。


---

## User <sup>2026/8/5 22:26:21</sup>

修复后可以正常推理；部署边云dp=2并行场景，边侧是2卡，云侧就是两台各自8卡的服务器，相比于非边云的dp=2并行场景，推理性能很差，从实现流程上分析一下性能瓶颈点，可以增加error级别执行时间日志确认，我可以跑两种场景的数据


---

## Assistant <sup>2026/8/5 22:27:39</sup>


---

## Assistant <sup>2026/8/5 22:27:41</sup>

修复验证通过，很好。现在分析边云 DP=2 的性能瓶颈。先理解架构:边侧 2 卡做 head/tail 段,云侧两台 8 卡服务器做 middle 段,每步要 edge↔cloud 跨节点传中间张量,且 prefill 是 one-real-one-dummy。这相比非边云(每台服务器本地跑完整模型)多了跨节点传输和分段流水线开销。

让我并行深入三部分实现流程,定位耗时点和打桩位置。


---

## Assistant <sup>2026/8/5 22:27:47</sup>

<details>
<summary>Tool: <code>Agent</code></summary>

```json
{
  "description": "Map edge-side step flow",
  "prompt": "分析边云 DP=2 场景下**边侧 EngineCore** 的每步执行流程与耗时点。代码在 `c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py`。\n\n场景：边云 PD-separation，DP=2，边侧 2 卡做 head/tail 段，云侧两台 8 卡做 middle 段。对比非边云 DP=2（每台服务器本地跑完整模型，无跨节点传输）。\n\n请精读该文件，梳理边侧 EngineCore 一个 step 的完整流程（busy loop / _patched_step / 协调），重点定位以下环节的代码位置（file:line）和调用关系：\n1. 跨 DP batch_type 协调 `_coordinate_bt`（all_reduce）——每步是否都做？耗时点。\n2. `_drain_pd_channel_inbox`（云侧返回的 tail 批次入队）——是否阻塞/轮询？\n3. 调度 `_schedule_target` / `_intended_batch_type`。\n4. head 段执行（execute_model，PF_FIRST/DF_FIRST）+ `_maybe_publish_pre_out`（向云侧发 ZMQ + hidden channel）。\n5. 等待云侧返回（POST_OUT）——阻塞在 `future.result()` 还是别的？等待期间能否做其他事？\n6. tail 段执行（PF_LAST/DECODE_LAST）。\n7. dummy 相关（`_patched_execute_dummy_batch`、`_publish_pd_dummy_zmq`）。\n8. 协调 Rule1/Rule2 的 defer 是否增加额外空转步。\n\n对每个环节，指出：是否是潜在性能瓶颈（相对非边云新增的开销），以及**适合插入 error 级耗时日志的具体位置（file:line + 测什么）**。特别关注：跨节点传输前后、等待云侧返回的空等、协调 all_reduce、defer 空转步。\n\n只读代码分析，不要修改文件。输出：流程图式步骤列表（含 file:line）、瓶颈点排序、建议的耗时日志打桩点列表。",
  "run_in_background": false,
  "subagent_type": "general-purpose"
}
```

</details>


---

## Assistant <sup>2026/8/5 22:27:53</sup>

<details>
<summary>Tool: <code>Agent</code></summary>

```json
{
  "description": "Map cloud-side flow",
  "prompt": "分析边云 DP=2 场景下**云侧** 的执行流程与耗时点。代码在 `c:/theroadtofree_github/vllm-ascend/vllm_ascend/` 目录下，重点 `core/passive_core.py`（云侧 PassiveEngineCore）、`worker/worker.py`（云侧 worker 的 `_execute_model_cloud` 及相关）、`worker/model_runner_v1.py`（middle 段执行）。\n\n场景：边云 PD-separation，DP=2，边侧 2 卡做 head/tail，云侧两台 8 卡服务器做 middle 段（MoE）。云侧是 PassiveEngineCore（被动接收边侧发来的批次）。\n\n请精读相关代码，梳理云侧处理一个 middle 段批次的完整流程，重点定位（file:line）：\n1. 云侧如何接收边侧的 SchedulerOutput（ZMQ channel recv / poll 机制）——是否忙等？延迟？\n2. 云侧如何接收 hidden tensors（`edge_cloud_irecv_tensor_dict` / NPU P2P）——与 ZMQ metadata 的时序关系。\n3. 云侧 middle 段 forward 执行（`_execute_model_cloud`、is_pd_dummy 分支、segment_c）。\n4. 云侧如何把结果发回边侧（`edge_cloud_isend_tensor_dict` + ZMQ post_out）。\n5. 云侧是否每步也做 `_sync_metadata_across_dp`（跨 DP all_reduce）？做几次（middle 段前后）？\n6. 云侧 dummy-middle 的处理（is_pd_dummy）——是否仍有开销？\n7. 云侧 PassiveEngineCore 的主循环开销、与边侧的同步点。\n\n对每个环节指出：是否潜在性能瓶颈，以及**适合插入 error 级耗时日志的位置（file:line + 测什么）**。特别关注：接收轮询延迟、irecv/isend 跨节点传输耗时、middle forward 耗时、sync_metadata all_reduce、dummy-middle 开销、云侧空等边侧。\n\n只读代码分析，不要修改文件。输出：云侧流程步骤（含 file:line）、瓶颈点排序、建议耗时日志打桩点列表。",
  "run_in_background": false,
  "subagent_type": "general-purpose"
}
```

</details>


---

## Assistant <sup>2026/8/5 22:28:00</sup>

<details>
<summary>Tool: <code>Agent</code></summary>

```json
{
  "description": "Map transfer mechanism and segments",
  "prompt": "分析边云 DP=2 场景下**跨节点传输机制与分段/集合通信结构**。代码在 `c:/theroadtofree_github/` 下：\n- `vllm/vllm/distributed/parallel_state.py`（`edge_cloud_isend_tensor_dict` / `edge_cloud_irecv_tensor_dict` / hidden channel / tensor meta）\n- `vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py`（`PPSchedulerZmqChannel` 的 publish/recv）\n- `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`（head/middle/tail 分段、layer_slice、EP all-toall、每段 sync_metadata）\n\n场景：边云 PD-separation，DP=2，边侧 2 卡，云侧两台 8 卡。全 MoE + EP，head/tail/middle 各段都走跨 DP EP all-toall。\n\n请精读代码，回答：\n1. **传输什么、怎么传**：ZMQ channel 传的是 SchedulerOutput（metadata）还是含 tensor？hidden tensors 走 `edge_cloud_isend/irecv_tensor_dict`（NPU hccl P2P）？两者如何配合（metadata 先到，tensor 后到？）。\n2. **每次 forward 要几次跨节点传输**：head->middle、middle->tail 各几次？是否每层都传还是按段传一次？tensor 大小量级（hidden_states 维度）。\n3. **序列化开销**：ZMQ 是否 cloudpickle 序列化 tensor？还是只传 metadata（tensor 走 NPU P2P）？\n4. **EP all-toall 结构**：EP group 怎么构成？边侧 2 卡 + 云侧 16 卡是否同一 EP group？all-toall 是否跨 DP（边↔云）？每段（head/middle/tail）各做几次 all-toall（每 MoE 层一次）？\n5. **每段 sync_metadata**：head/middle/tail 各做几次 `_sync_metadata_across_dp`？总共每步几次跨 DP all_reduce？\n6. **layer_slice / head_k / tail_k**：分段层数怎么定？边侧跑几层，云侧跑几层？\n\n对每个发现指出是否性能瓶颈，以及**适合插入 error 级耗时日志的位置（file:line + 测什么）**。特别关注：tensor 传输耗时（isend/irecv 前后）、序列化耗时、EP all-toall 耗时、每段 sync_metadata 耗时、传输次数总量。\n\n只读代码分析，不要修改文件。输出：传输机制说明、每次 forward 的传输/集合通信次数清单、瓶颈点排序、建议耗时日志打桩点列表（file:line）。",
  "run_in_background": false,
  "subagent_type": "general-purpose"
}
```

</details>


---

## User <sup>2026/8/5 22:31:50</sup>

<details>
<summary>Tool Result</summary>

```
我现在已经完整掌握了这三个文件及其相互作用的情况。以下是我的分析。

---

# 边云 DP=2 跨节点传输机制与集合通信结构分析

## 场景拓扑确认

DP=2，边侧 2 卡，云侧两台 8 卡（共 16 卡）。标准（非 shared-model）edge-cloud 拓扑：
- `edge_npu_count=1` per DP instance，`cloud_npu_count=8` per DP instance
- `world_size_per_instance = 1+8 = 9`，`world_size = 18`
- rank 布局：DP0={edge:0, cloud:1..8}，DP1={edge:9, cloud:10..17}

---

## 1. 传输什么、怎么传

系统使用**双轨传输**：ZMQ 传 metadata（SchedulerOutput），NPU HCCL P2P 传 tensor（hidden_states + residual）。

### ZMQ channel — 只传 SchedulerOutput（metadata，无 tensor）

`PPSchedulerZmqChannel`（`passive_core.py:298`）拥有一个 publisher（PUSH）+ 一个 subscriber（PULL），各自跑在独立后台线程上。

- **publish**（`passive_core.py:354` → `PPSchedulerZmqPublisher.publish` at `:131`）：调用方线程只把 `SchedulerOutput` 对象 `put_nowait` 进 `queue.Queue`，**不阻塞 scheduler**。
- 后台线程 `_publisher_thread`（`:146`）从队列取出，`pickle.dumps(scheduler_output, protocol=HIGHEST_PROTOCOL)`（`:166`），`send_multipart((seq_bytes, data))`。
- **consume**（`passive_core.py:362` → subscriber `:278`）：后台线程 `pickle.loads(data)`（`:23
...(truncated)
```

</details>


---

## User <sup>2026/8/5 22:31:56</sup>

<details>
<summary>Tool Result</summary>

```
我已经完整梳理了整个流程。让我来整理一下分析结果。

---

## 边侧 EngineCore 单步执行流程与耗时点分析

代码主文件：`c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py`
关联文件：`vllm_ascend/patch/platform/patch_balance_schedule.py`（外层 busy loop）、`vllm_ascend/v1/engine/passive_core.py`（ZMQ channel）、`vllm_ascend/worker/worker.py`（head/tail 段实际执行 + 跨节点收发）、`vllm_ascend/core/pd_separated_scheduler.py`（`_intended_batch_type`/`_schedule_target`）。

### 一、调用关系总览（哪条 path 是热点）

边云 DP=2 走的是 **`_patched_step_with_batch_queue` 的 `_dp_parallel=True` 分支**（`patch_engine_core.py:1009`），不是 `_patched_step`（后者是 DP=1 / 非 batch_queue 路径，`patch_engine_core.py:823`）。`_dp_parallel` 在 `data_parallel_size>1` 时置 True 并缓存（`patch_engine_core.py:901-913`）。

外层驱动是 `BalanceDPEngineCoreProc.run_busy_loop`（`patch_balance_schedule.py:623`），每轮：
```
_process_input_queue()          # patch_engine_core.py:1328
_process_engine_step()          # patch_engine_core.py:1308 -> step_fn() = step_with_batch_queue
  └─ step_with_batch_queue()    # pat
...(truncated)
```

</details>


---

## User <sup>2026/8/5 22:36:29</sup>

<details>
<summary>Tool Result</summary>

```
我已经掌握了所需的所有信息。以下是综合分析。

---

# 云侧（PassiveEngineCore）DP=2 middle 段执行流程与耗时点分析

## 一、云侧处理一个 middle 段批次的完整流程

### 步骤 0：ZMQ 接收边侧 SchedulerOutput（后台线程）

- `c:/theroadtofree_github/vllm-ascend/vllm_ascend/v1/engine/passive_core.py:229-264` — `PPSchedulerZmqSubscriber._subscriber_thread`：ZMQ PULL socket **`poll(timeout=100ms)`**（line 232），非忙等；收到后 `pickle.loads` 反序列化，append 到 `_received_outputs`。
- `c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py:251-267` — `PassiveScheduler._subscriber_loop`：另一个后台线程调用 `pp_subscriber.consume_new_outputs()`，空时 `_shutdown_event.wait(0.001)`（1ms 自旋），非空时 `self._inbox.put(...)`。
- **关键点**：ZMQ 控制面 metadata 走 PULL socket（100ms poll）；hidden tensors 走 NPU P2P（HCCL irecv）。两者**时序独立**，靠 `head_token` 关联。云侧拿到 metadata 后即可触发 `cloud_prepare_early` + early-irecv，**不必等 hidden 到达**。

### 步骤 1：PassiveEC.step() 主循环单 tick

`passive_core.py:608-798`：

1. **`_drain_worker_completion_acks()`**（line 623）— 非阻塞（timeout=0） drain `response_mqs`，看上一个 middle 是否完成，完成则
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 22:37:48</sup>


---

## Assistant <sup>2026/8/5 22:37:48</sup>

三份分析已汇齐。让我先验证几个关键打桩点的准确位置(云侧 worker 收发、PassiveScheduler 的 3 次 all_reduce、irecv wait 点),再统一加日志。


---

## Assistant <sup>2026/8/5 22:37:49</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/worker.py",
  "limit": 60,
  "offset": 955
}
```

</details>


---

## User <sup>2026/8/5 22:37:49</sup>

<details>
<summary>Tool Result</summary>

```
955	
956	        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
957	            return output
958	
959	        assert isinstance(output, IntermediateTensors)
960	        # Edge-cloud with heterogeneous SP: aggregate SP shards to full
961	        # sequence before cross-PP send so cloud can re-chunk by its SP.
962	        if enable_sp() and (self.model_runner.edge_cloud_cfg.mode != "embedding_only"
963	            or not self.model_runner.supports_mm_inputs):
964	            _gathered = self._all_gather_tensor_dict(output.tensors)
965	        else:
966	            _gathered = output.tensors
967	        if get_pp_group().world_size == 2:
968	            channel = self._hidden_channel_for(scheduler_output)
969	            self._record_pp_send_work(
970	                edge_cloud_send_tensor_dict(_gathered, channel=channel,
971	                                            num_tokens=scheduler_output.total_num_scheduled_tokens),
972	                channel=chan
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 22:37:49</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/worker.py",
  "limit": 60,
  "offset": 1020
}
```

</details>


---

## User <sup>2026/8/5 22:37:49</sup>

<details>
<summary>Tool Result</summary>

```
1020	        scheduler_output: "SchedulerOutput",
1021	        layer_slice_info: Any,
1022	    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
1023	        edge_sp = enable_sp()
1024	        edge_merge = get_edge_cloud_tensor_meta().merge_payload
1025	        """Edge tail segment (PL/DL): recv -> segment_e -> return output."""
1026	        logger.info(f"Execute model, batch_type: {scheduler_output.batch_type}")
1027	        channel = self._hidden_channel_for(scheduler_output)
1028	        _hang_tail_rank = getattr(self.model_runner, "dp_rank", "?")
1029	        logger.info("[HANG] edge tail recv ENTER: dp_rank=%s channel=%s batch_type=%s",
1030	                    _hang_tail_rank, channel.value, scheduler_output.batch_type)
1031	        import sys as _hang_sys
1032	        _hang_sys.stderr.flush()
1033	        tensor_dict, comm_handles, comm_postprocess = edge_cloud_broadcast_recv(
1034	            num_tokens=scheduler_output.total_num_scheduled_tokens,
1035	            channel
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 22:37:50</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/worker.py",
  "limit": 70,
  "offset": 1175
}
```

</details>


---

## User <sup>2026/8/5 22:37:50</sup>

<details>
<summary>Tool Result</summary>

```
1175	                # distributed rank at in-group rank 0; the cloud first-worker of
1176	                # each dp_rank must receive from that rank (src=0).  In the
1177	                # standard (non-shared-model) topology src=None suffices: it
1178	                # resolves to the implicit "previous PP rank" which IS the edge.
1179	                _recv_src = 0 if self.parallel_config.is_shared_model_edge else None
1180	                tensor_dict, comm_handles, comm_postprocess = edge_cloud_broadcast_recv(
1181	                    num_tokens=scheduler_output.total_num_scheduled_tokens,
1182	                    channel=channel,
1183	                    sp_chunk=do_sp_chunk and merge_payload,
1184	                    src=_recv_src,
1185	                )
1186	                logger.info(f"Received intermediate tensors from edge, hidden_channel={channel.value}")
1187	
1188	                if do_sp_chunk and not merge_payload:
1189	                    tensor_dict = {
1190	          
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 22:38:19</sup>


---

## Assistant <sup>2026/8/5 22:38:21</sup>

我看到边缘/云端路径中已经存在 `[HANG]`/`[PP-EVT]` 日志。`irecv` 的 `wait` 操作实际上发生在稍后的 `AsyncIntermediateTensors.wait_for_comm` 中（`recv` 调用只是发起了一个异步 `irecv`）。让我验证一下 `PassiveScheduler` 的 3 个 `all_reduces` 并找到 `wait_for_comm`，以便我能正确地对实际的传输 `wait` 进行埋点。


---

## Assistant <sup>2026/8/5 22:38:21</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py",
  "limit": 110,
  "offset": 910
}
```

</details>


---

## User <sup>2026/8/5 22:38:21</sup>

<details>
<summary>Tool Result</summary>

```
910	        # 5 fields: ready_prefills, ready_decodes, ready_pdmixes,
911	        #           _active_prefill_slices, has _active_sliced_prefill
912	        local = [
913	            len(self.ready_prefills),
914	            len(self.ready_decodes),
915	            len(self.ready_pdmixes),
916	            len(self._active_prefill_slices),
917	            1 if self._active_sliced_prefill is not None else 0,
918	        ]
919	
920	        # Each rank writes to its own row; SUM leaves values independent.
921	        tensor = torch.zeros(_dp_size * 5, dtype=torch.int32, device="cpu")
922	        base = _dp_rank * 5
923	        for i, v in enumerate(local):
924	            tensor[base + i] = v
925	        dist.all_reduce(tensor, op=dist.ReduceOp.SUM,
926	                         group=self.dp_coord_group)
927	
928	        # Verify all rank values are identical for each of the 5 queues.
929	        all_match = True
930	        for q in range(5):
931	            vals = [int(tensor[r * 5 + q].
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 22:38:23</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== AsyncIntermediateTensors / wait_for_comm location ===\" && grep -rn \"class AsyncIntermediateTensors\\|def wait_for_comm\\|def tensors\\|wait_for_comm\" vllm-ascend/vllm_ascend/ vllm/vllm/v1/worker/ 2>/dev/null | head -15 && echo \"=== passive_scheduler all_reduce sites ===\" && grep -n \"all_reduce\\|dist\\.all_reduce\\|cpu_group\\|def sync_queue_state\\|def _schedule_expect_alternation\\|_replay_by_deltas\" vllm-ascend/vllm_ascend/core/passive_scheduler.py | head -25",
  "description": "Find wait_for_comm and passive_scheduler all_reduce sites"
}
```

</details>


---

## User <sup>2026/8/5 22:38:25</sup>

<details>
<summary>Tool Result</summary>

```
=== AsyncIntermediateTensors / wait_for_comm location ===
vllm-ascend/vllm_ascend/distributed/parallel_state.py:63:# it runs inside execute_model's wait_for_comm() on all TP ranks
vllm-ascend/vllm_ascend/distributed/parallel_state.py:1405:        # waited on by AsyncIntermediateTensors.wait_for_comm().
vllm-ascend/vllm_ascend/worker/worker.py:208:        # AsyncIntermediateTensors and runs wait_for_comm() (which both
vllm-ascend/vllm_ascend/worker/worker.py:1137:            # done by execute_model's wait_for_comm() on the busy_loop thread.
vllm-ascend/vllm_ascend/worker/worker.py:1150:                # wait_for_comm() runs implicitly on first .tensors access
vllm/vllm/v1/worker/gpu_worker.py:73:class AsyncIntermediateTensors(IntermediateTensors):
vllm/vllm/v1/worker/gpu_worker.py:87:    def wait_for_comm(self) -> None:
vllm/vllm/v1/worker/gpu_worker.py:101:            object.__getattribute__(self, "wait_for_comm")()
=== passive_scheduler all_reduce sites ===
151:        # ready_prefill
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 22:38:37</sup>


---

## Assistant <sup>2026/8/5 22:38:39</sup>

已验证：`PassiveScheduler` 在 `dp_coord_group` (跨云 DP) 上执行 3 次 `all_reduce` (925/979/1009 行)，且实际的 `irecv` 等待位于 [gpu_worker.py:87](vllm/vllm/v1/worker/gpu_worker.py#L87) 的 `wait_for_comm`。让我读取一下代码，以便对传输等待进行埋点。


---

## Assistant <sup>2026/8/5 22:38:39</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm/vllm/v1/worker/gpu_worker.py",
  "limit": 40,
  "offset": 73
}
```

</details>


---

## User <sup>2026/8/5 22:38:39</sup>

<details>
<summary>Tool Result</summary>

```
73	class AsyncIntermediateTensors(IntermediateTensors):
74	    """IntermediateTensors with lazy comm synchronization"""
75	
76	    def __init__(
77	        self,
78	        tensors: dict[str, torch.Tensor],
79	        comm_handles: list[Handle] | None = None,
80	        comm_postprocess: list[Callable[[], None]] | None = None,
81	    ) -> None:
82	        super().__init__(tensors)
83	        self._comm_handles = comm_handles
84	        self._comm_postprocess = comm_postprocess
85	        self._comm_waited = False
86	
87	    def wait_for_comm(self) -> None:
88	        if self._comm_waited:
89	            return
90	        if self._comm_handles:
91	            for handle in self._comm_handles:
92	                handle.wait()
93	        if self._comm_postprocess:
94	            for fn in self._comm_postprocess:
95	                fn()
96	        self._comm_waited = True
97	
98	    def __getattribute__(self, name: str):
99	        # ensure `.tensors` is ready before use
100	        if nam
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/5 22:39:59</sup>


---

## Assistant <sup>2026/8/5 22:39:59</sup>

所有插入点已验证。让我检查一下 `gpu_worker.py` 中的 logger 导入，然后添加一套统一的时间日志。


---

## Assistant <sup>2026/8/5 22:40:01</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && echo \"=== gpu_worker.py logger/time imports ===\" && sed -n '1,30p' vllm/vllm/v1/worker/gpu_worker.py | grep -n \"import\\|logger\" && echo \"=== passive_scheduler logger ===\" && grep -n \"^import logging\\|^from.*logging\\|logger = \\|^import time\\|import time\" vllm-ascend/vllm_ascend/core/passive_scheduler.py | head && echo \"=== patch_engine_core logger/time ===\" && grep -n \"^import time\\|logger = \\|^logger\\|vllm_logger = \" vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py | head && echo \"=== model_runner logger/time ===\" && grep -n \"^import time\\|logger = \" vllm-ascend/vllm_ascend/worker/model_runner_v1.py | head",
  "description": "Check logger and time imports in target files"
}
```

</details>


---

## User <sup>2026/8/5 22:40:04</sup>

<details>
<summary>Tool Result</summary>

```
=== gpu_worker.py logger/time imports ===
5:import gc
6:import os
7:from collections.abc import Callable
8:from contextlib import AbstractContextManager, nullcontext
9:from datetime import timedelta
10:from types import NoneType
11:from typing import TYPE_CHECKING, Any
13:import numpy as np
14:import torch
15:import torch.nn as nn
17:import vllm.envs as envs
18:from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
19:from vllm.config.compilation import CompilationMode
20:from vllm.distributed import (
25:from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
26:from vllm.distributed.eplb.eplb_utils import override_envs_for_eplb
27:from vllm.distributed.kv_transfer import (
=== passive_scheduler logger ===
21:import time
=== patch_engine_core logger/time ===
79:logger = init_logger(__name__)
1304:import time as _time  # noqa: E402
=== model_runner logger/time ===
23:import time
```

</details>


---

## Assistant <sup>2026/8/5 22:40:20</sup>


---

## Assistant <sup>2026/8/5 22:40:21</sup>

`gpu_worker.py` lacks a logger/time import — I'll use inline imports there. Others have what's needed. Let me add the focused timing-log set. Starting with the edge-side coordination/blocking points in [patch_engine_core.py](vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py).


---

## Assistant <sup>2026/8/5 22:40:22</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py",
  "old_string": "    torch.distributed.all_reduce(tensor, group=dp_group)\n    bt_ids = [int(tensor[r].item()) for r in range(dp_size)]",
  "new_string": "    import time as _ec_perf_time\n    _ar_t0 = _ec_perf_time.monotonic()\n    torch.distributed.all_reduce(tensor, group=dp_group)\n    _ar_dt_ms = (_ec_perf_time.monotonic() - _ar_t0) * 1000\n    bt_ids = [int(tensor[r].item()) for r in range(dp_size)]\n    logger.error(\n        \"[EC-PERF][COORD-BT] dp_rank=%s cnt=%s all_reduce=%.3fms bt_ids=%s\",\n        dp_rank, _cnt, _ar_dt_ms, bt_ids,\n    )"
}
```

</details>


---

## User <sup>2026/8/5 22:40:23</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 22:40:29</sup>


---

## Assistant <sup>2026/8/5 22:40:30</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py",
  "old_string": "        model_output = future.result()\n        if model_output is None:",
  "new_string": "        import time as _ec_perf_time\n        _fr_t0 = _ec_perf_time.monotonic()\n        model_output = future.result()\n        _fr_dt_ms = (_ec_perf_time.monotonic() - _fr_t0) * 1000\n        vllm_logger.error(\n            \"[EC-PERF][EC-BLOCK] dp_rank=%s bt=%s tokens=%s future.result=%.3fms\",\n            self.vllm_config.parallel_config.data_parallel_rank,\n            bt.value if bt else \"N/A\",\n            scheduler_output.total_num_scheduled_tokens, _fr_dt_ms,\n        )\n        if model_output is None:"
}
```

</details>


---

## User <sup>2026/8/5 22:40:30</sup>

<details>
<summary>Tool Result</summary>

```
<tool_use_error>Found 2 matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. To replace only one occurrence, please provide more context to uniquely identify the instance.
String:         model_output = future.result()
        if model_output is None:</tool_use_error>
```

</details>


---

## Assistant <sup>2026/8/5 22:40:35</sup>


---

## Assistant <sup>2026/8/5 22:40:36</sup>

需要更多上下文来消除歧义。让我根据周围的 `with` 代码块将其包含进去。


---

## Assistant <sup>2026/8/5 22:40:37</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py",
  "old_string": "        self.log_iteration_details(scheduler_output),\n    ):\n        model_output = future.result()\n        if model_output is None:",
  "new_string": "        self.log_iteration_details(scheduler_output),\n    ):\n        import time as _ec_perf_time\n        _fr_t0 = _ec_perf_time.monotonic()\n        model_output = future.result()\n        _fr_dt_ms = (_ec_perf_time.monotonic() - _fr_t0) * 1000\n        vllm_logger.error(\n            \"[EC-PERF][EC-BLOCK] dp_rank=%s bt=%s tokens=%s future.result=%.3fms\",\n            self.vllm_config.parallel_config.data_parallel_rank,\n            bt.value if bt else \"N/A\",\n            scheduler_output.total_num_scheduled_tokens, _fr_dt_ms,\n        )\n        if model_output is None:"
}
```

</details>


---

## User <sup>2026/8/5 22:40:37</sup>

<details>
<summary>Tool Result</summary>

```
<tool_use_error>Found 2 matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. To replace only one occurrence, please provide more context to uniquely identify the instance.
String:         self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:</tool_use_error>
```

</details>


---

## Assistant <sup>2026/8/5 22:40:47</sup>


---

## Assistant <sup>2026/8/5 22:40:47</sup>

存在两处相同的 `future.result()` 代码块（DP=1 路径和 batch_queue 路径）。让我利用 batch_queue 路径中特有的后续行来进行区分。


---

## Assistant <sup>2026/8/5 22:40:49</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py",
  "old_string": "        model_output = future.result()\n        if model_output is None:\n            exec_model_fut.result()\n            raise RuntimeError(\"unexpected error\")",
  "new_string": "        import time as _ec_perf_time\n        _fr_t0 = _ec_perf_time.monotonic()\n        model_output = future.result()\n        _fr_dt_ms = (_ec_perf_time.monotonic() - _fr_t0) * 1000\n        vllm_logger.error(\n            \"[EC-PERF][EC-BLOCK] dp_rank=%s bt=%s tokens=%s future.result=%.3fms\",\n            self.vllm_config.parallel_config.data_parallel_rank,\n            bt.value if bt else \"N/A\",\n            scheduler_output.total_num_scheduled_tokens, _fr_dt_ms,\n        )\n        if model_output is None:\n            exec_model_fut.result()\n            raise RuntimeError(\"unexpected error\")"
}
```

</details>


---

## User <sup>2026/8/5 22:40:50</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 22:40:56</sup>


---

## Assistant <sup>2026/8/5 22:40:56</sup>

现在是 Rule1/Rule2 的延迟事件：


---

## Assistant <sup>2026/8/5 22:40:59</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py",
  "old_string": "        if _DL in bt_ids:\n            # Rule 2: let the DP with a ready DL run it (real); the\n            # DF-ready peer runs a dummy DL so both align on DF next step.\n            winner_id = _DL\n        elif _DF in bt_ids and _E in bt_ids:\n            # Rule 1: defer DF when the EMPTY peer is waiting on its DL\n            # (has decode work). If the EMPTY peer is truly idle, fall\n            # through to the default DF so the dummy pairs the a2a.\n            if any(bt_ids[r] == _E and waitings[r] for r in range(dp_size)):\n                winner_id = _E",
  "new_string": "        if _DL in bt_ids:\n            # Rule 2: let the DP with a ready DL run it (real); the\n            # DF-ready peer runs a dummy DL so both align on DF next step.\n            winner_id = _DL\n            logger.error(\n                \"[EC-PERF][DEFER] dp_rank=%s cnt=%s rule=2(DECODE_LAST) \"\n                \"bt_ids=%s waitings=%s -> peer dummy DL step\",\n                dp_rank, _cnt, bt_ids, waitings,\n            )\n        elif _DF in bt_ids and _E in bt_ids:\n            # Rule 1: defer DF when the EMPTY peer is waiting on its DL\n            # (has decode work). If the EMPTY peer is truly idle, fall\n            # through to the default DF so the dummy pairs the a2a.\n            if any(bt_ids[r] == _E and waitings[r] for r in range(dp_size)):\n                winner_id = _E\n                logger.error(\n                    \"[EC-PERF][DEFER] dp_rank=%s cnt=%s rule=1(EMPTY-defer) \"\n                    \"bt_ids=%s waitings=%s -> DF deferred 1 step (no compute)\",\n                    dp_rank, _cnt, bt_ids, waitings,\n                )"
}
```

</details>


---

## User <sup>2026/8/5 22:41:00</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 22:41:07</sup>


---

## Assistant <sup>2026/8/5 22:41:07</sup>

现在 per-segment 的 `sync_metadata` all_reduce：


---

## Assistant <sup>2026/8/5 22:41:08</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
  "old_string": "        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)\n\n        # Extract peer's batch_type_id (for DUMMY to match real's segment)\n        self._peer_batch_type_id = int(packed_tensor[2, 1 - self.dp_rank].item())",
  "new_string": "        import time as _ec_perf_time\n        _sm_t0 = _ec_perf_time.monotonic()\n        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)\n        _sm_dt_ms = (_ec_perf_time.monotonic() - _sm_t0) * 1000\n\n        # Extract peer's batch_type_id (for DUMMY to match real's segment)\n        self._peer_batch_type_id = int(packed_tensor[2, 1 - self.dp_rank].item())\n        logger.error(\n            \"[EC-PERF][SYNC-META] dp_rank=%s my_bt=%s peer_bt=%s \"\n            \"num_tokens=%s all_reduce=%.3fms\",\n            self.dp_rank, self._dp_batch_type_id,\n            self._peer_batch_type_id, num_tokens, _sm_dt_ms,\n        )"
}
```

</details>


---

## User <sup>2026/8/5 22:41:09</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/worker/model_runner_v1.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 22:41:16</sup>


---

## Assistant <sup>2026/8/5 22:41:16</sup>

现在是 `gpu_worker.py` 中跨节点 tensor 的等待点：


---

## Assistant <sup>2026/8/5 22:41:18</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm/vllm/v1/worker/gpu_worker.py",
  "old_string": "    def wait_for_comm(self) -> None:\n        if self._comm_waited:\n            return\n        if self._comm_handles:\n            for handle in self._comm_handles:\n                handle.wait()\n        if self._comm_postprocess:\n            for fn in self._comm_postprocess:\n                fn()\n        self._comm_waited = True",
  "new_string": "    def wait_for_comm(self) -> None:\n        if self._comm_waited:\n            return\n        import time as _ec_perf_time\n        import logging as _ec_perf_logging\n        _wc_t0 = _ec_perf_time.monotonic()\n        if self._comm_handles:\n            for handle in self._comm_handles:\n                handle.wait()\n        if self._comm_postprocess:\n            for fn in self._comm_postprocess:\n                fn()\n        _wc_dt_ms = (_ec_perf_time.monotonic() - _wc_t0) * 1000\n        _ec_perf_logging.getLogger(\"vllm\").error(\n            \"[EC-PERF][IRECV-WAIT] cross-node tensor wait=%.3fms handles=%s\",\n            _wc_dt_ms, len(self._comm_handles or []),\n        )\n        self._comm_waited = True"
}
```

</details>


---

## User <sup>2026/8/5 22:41:19</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm/vllm/v1/worker/gpu_worker.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 22:41:32</sup>


---

## Assistant <sup>2026/8/5 22:41:32</sup>

现在，`PassiveScheduler` 中的 3 个云端协调 `all_reduce`（每个步骤中最大的一个新增开销集合）：


---

## Assistant <sup>2026/8/5 22:41:34</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py",
  "old_string": "        dist.all_reduce(tensor, op=dist.ReduceOp.SUM,\n                         group=self.dp_coord_group)\n\n        # Verify all rank values are identical for each of the 5 queues.",
  "new_string": "        _cc_t0 = time.monotonic()\n        dist.all_reduce(tensor, op=dist.ReduceOp.SUM,\n                         group=self.dp_coord_group)\n        logger.error(\n            \"[EC-PERF][CLOUD-COORD] dp_rank=%s phase=sync_queue all_reduce=%.3fms\",\n            self.vllm_config.parallel_config.data_parallel_rank,\n            (time.monotonic() - _cc_t0) * 1000,\n        )\n\n        # Verify all rank values are identical for each of the 5 queues."
}
```

</details>


---

## User <sup>2026/8/5 22:41:35</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 22:41:41</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py",
  "old_string": "        _sync = torch.tensor([_local_intent], dtype=torch.int32)\n        dist.all_reduce(_sync, op=dist.ReduceOp.MAX,\n                        group=self.dp_coord_group)\n        self._coordinated_total_slices = int(_sync.item()) or None",
  "new_string": "        _sync = torch.tensor([_local_intent], dtype=torch.int32)\n        _cc_t0 = time.monotonic()\n        dist.all_reduce(_sync, op=dist.ReduceOp.MAX,\n                        group=self.dp_coord_group)\n        logger.error(\n            \"[EC-PERF][CLOUD-COORD] dp_rank=%s phase=intent_max all_reduce=%.3fms\",\n            _dp_rank, (time.monotonic() - _cc_t0) * 1000,\n        )\n        self._coordinated_total_slices = int(_sync.item()) or None"
}
```

</details>


---

## User <sup>2026/8/5 22:41:42</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 22:41:46</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py",
  "old_string": "        dist.all_reduce(_tensor, op=dist.ReduceOp.SUM,\n                         group=self.dp_coord_group)\n        d_prefills, d_decodes, d_pdmixes, d_slices = _tensor.tolist()",
  "new_string": "        _cc_t0 = time.monotonic()\n        dist.all_reduce(_tensor, op=dist.ReduceOp.SUM,\n                         group=self.dp_coord_group)\n        logger.error(\n            \"[EC-PERF][CLOUD-COORD] dp_rank=%s phase=deltas_sum all_reduce=%.3fms\",\n            _dp_rank, (time.monotonic() - _cc_t0) * 1000,\n        )\n        d_prefills, d_decodes, d_pdmixes, d_slices = _tensor.tolist()"
}
```

</details>


---

## User <sup>2026/8/5 22:41:47</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/5 22:41:52</sup>


---

## Assistant <sup>2026/8/5 22:41:52</sup>

已添加所有 6 个埋点点。我来对修改后的文件进行语法检查以验证其准确性。


---

## Assistant <sup>2026/8/5 22:41:53</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && for f in vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py vllm-ascend/vllm_ascend/worker/model_runner_v1.py vllm-ascend/vllm_ascend/core/passive_scheduler.py vllm/vllm/v1/worker/gpu_worker.py; do python -m py_compile \"$f\" 2>&1 && echo \"OK: $f\" || echo \"FAIL: $f\"; done",
  "description": "Syntax-check modified files"
}
```

</details>


---

## User <sup>2026/8/5 22:43:16</sup>

<details>
<summary>Tool Result</summary>

```
OK: vllm-ascend/vllm_ascend/patch/platform/patch_engine_core.py
OK: vllm-ascend/vllm_ascend/worker/model_runner_v1.py
OK: vllm-ascend/vllm_ascend/core/passive_scheduler.py
OK: vllm/vllm/v1/worker/gpu_worker.py
```

</details>


---

## Assistant <sup>2026/8/5 22:43:46</sup>


---

## Assistant <sup>2026/8/5 22:44:08</sup>

语法全部通过。下面是完整的流程分析与已加入的耗时日志说明。

## 一、边云 DP=2 vs 非边云 DP=2 流程对比

**非边云**：每台服务器本地跑完整模型,一个 forward 内部层间流水,每步仅 1 次 `sync_metadata` all_reduce,`has_unfinished` 每 32 步才同步一次。

**边云**(边 2 卡 / 云 2 台×8 卡,全 MoE+EP,head_k=tail_k=1)每个真实 batch 变成**三段串行流水**:

```
边侧 head(1层) ─isend→ [跨节点] →云侧 irecv_wait → 云 middle(N-2层) ─isend→ [跨节点] →边侧 irecv_wait → 边 tail(1层)+sample
   │ZMQ PRE_OUT(metadata)→│           │ZMQ POST_OUT←│
```

并且**两 DP 强制同 batch_type**(EP all-toall 跨 DP 配对),prefill 阶段一个 real 一个 dummy。

## 二、性能瓶颈点(相对非边云的新增开销,按疑似严重度排序)

| 级别 | 瓶颈 | 位置 | 为什么变慢 |
|---|---|---|---|
| **P0** | **串行跨节点往返** | `worker.py:1033/1180` irecv + `gpu_worker.py:87` wait | 每个 forward 多 2 次跨物理机 tensor 传输,且 head→middle→tail 串行;decode(1 token)时往返延迟直接主导单步延迟。CHER 只覆盖 PREFILL_FIRST,DECODE_FIRST 仍同步等 |
| **P1** | **跨 DP all_reduce 泛滥** | 见下 | 非边云 ~1 次/步;边云每步 4~7+ 次 gloo CPU barrier(跨服务器),每次 0.1~1ms+,且慢方拖快方 |
| **P2** | **prefill one-real-one-dummy** | `_coordinate_bt` Rule | prefill 一个 DP real、一个 dummy,吞吐直接减半;Rule1/2 还会插入空转/defer 步 |
| **P3** | **云侧 PassiveScheduler 开销** | `passive_scheduler.py` | 非边云无此组件:每步 3 次 all_reduce + ZMQ poll(100ms) + 1ms idle sleep + 每 slice 一次 MQ 序列化 |
| **P4** | **小边侧** | edge 2 卡跑 head/tail | head_k=tail_k=1 仅 2 层,影响应有限 |

**P1 的 all_reduce 清单(每真实 forward)**:
- 边 EngineCore:`_coordinate_bt` **每步 1 次**(`patch_engine_core.py:781`,非边云 32 步 1 次)
- 云 PassiveScheduler:**每步 3 次**(`passive_scheduler.py` sync_queue + intent_max + deltas_sum,非边云无)
- model_runner:**每段 1 次** `sync_metadata`(head+middle+tail=3,layer slice 续段每片 +1;非边云 1 次/步)
- dummy 也做 1 次 sync(必须 1:1 配对)

→ 无 slice 时 real 侧 ≈ **7 次**跨 DP barrier/步,dummy 侧 ≈ 5 次。这是相对非边云最显著的同步放大。

注:云侧 middle 的 EP all-toall(16 卡)与非边云(同样 16 卡跨 2 服务器)规模相近,**不是差异点**;差异在传输+协调+串行+dummy。

## 三、已加入的 error 级耗时日志(统一 tag `[EC-PERF]`)

| Tag | 位置 | 测什么 | 何时触发 |
|---|---|---|---|
| `[EC-PERF][COORD-BT]` | `patch_engine_core.py:781` | 边侧协调 all_reduce 耗时 | 边云每步(非边云不触发) |
| `[EC-PERF][CLOUD-COORD]` | `passive_scheduler.py:925/979/1009` | 云侧 3 次 all_reduce(sync_queue/intent_max/deltas_sum) | 边云每步(非边云无 PassiveScheduler) |
| `[EC-PERF][SYNC-META]` | `model_runner_v1.py:1693` | 每段 sync_metadata all_reduce + my_bt/peer_bt/tokens | 两场景都触发(非边云 1/步,边云 3+/步) |
| `[EC-PERF][IRECV-WAIT]` | `gpu_worker.py:87` wait_for_comm | 跨节点 tensor 等待(=对端算+传输) | 边云每段收tensor(非边云不触发) |
| `[EC-PERF][EC-BLOCK]` | `patch_engine_core.py:1202` | 边 EngineCore 阻塞在 future.result(整轮云往返) | 边云每 batch(非边云不触发) |
| `[EC-PERF][DEFER]` | `patch_engine_core.py:799-808` | Rule1/Rule2 defer 事件(空转/推迟步) | 边云命中时(非边云不触发) |

## 四、如何跑数据对比

1. 两种场景跑**相同 workload**(建议先短:单请求 N token,够看清每步即可;长跑日志量大)。
2. `grep "\[EC-PERF\]"` 收集,按 tag 聚合:
   - **非边云**:基本只有 `[SYNC-META]`(1/步),其余不触发 → 证明这些开销是边云独有。
   - **边云**:逐 tag 算"每步总耗时贡献":
     - `Σ IRECV-WAIT` ≈ 跨节点串行等待(P0)
     - `Σ COORD-BT + Σ CLOUD-COORD + Σ SYNC-META` ≈ 协调 barrier 总开销(P1)
     - `EC-BLOCK` ≈ 单 batch 端到端阻塞(应 ≈ IRECV-WAIT + 云 middle + 传输),即用户感知的单步延迟
     - `DEFER` 计数 ≈ 浪费步数(P2)
3. 哪个 tag 占 `EC-BLOCK` 比例最大,即主瓶颈。

预判:边云的 `EC-BLOCK` 会远大于非边云单步,其中 `IRECV-WAIT`(串行往返)+ 三类 all_reduce 之和是大头;`DEFER` 在 prefill 期会频繁出现。跑完把两场景各 tag 的典型值/汇总发我,可进一步定位是传输带宽、协调 barrier 还是 dummy 占主导,再决定优化方向(如 decode 也上 CHER、减少云侧 all_reduce 次数、prefill 两 real 等)。

需要的话我可以再加"云 middle forward 纯计算耗时"和"边 head/tail forward 耗时"两个点,用于把 `EC-BLOCK` 进一步拆成"传输+协调 vs 纯计算"。


---

## User <sup>2026/8/6 09:16:07</sup>

<ide_opened_file>The user opened the file c:\theroadtofree_github\vllm-ascend\vllm_ascend\core\passive_scheduler.py in the IDE. This may or may not be related to the current task.</ide_opened_file>

没有请求时云侧就打印C:\theroadtofree_github\vllm-ascend\vllm_ascend\core\passive_scheduler.py的928，988，1023行，云侧调度能否在有请求时再执行


---

## Assistant <sup>2026/8/6 09:17:32</sup>


---

## Assistant <sup>2026/8/6 09:17:34</sup>

好的,这是个真实的空转问题:云侧 `PassiveScheduler` 在无请求时仍每步做 3 次 all_reduce(`sync_queue_state` + `intent_max` + `deltas_sum`),既刷日志又空耗。我先看下当前 `step()` 的调用结构和空闲判定逻辑,再决定怎么让"调度只在有请求时执行"且不破坏两云 DP 的 barrier 配对。


---

## Assistant <sup>2026/8/6 09:17:34</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py",
  "limit": 60,
  "offset": 885
}
```

</details>


---

## User <sup>2026/8/6 09:17:34</sup>

<details>
<summary>Tool Result</summary>

```
885	    # ------------------------------------------------------------------ #
886	
887	    def sync_queue_state(self) -> bool:
888	        """Sync queue lengths across cloud DPs via all_reduce.
889	
890	        Checks that ready_prefills, ready_decodes, ready_pdmixes,
891	        _active_prefill_slices, and _active_sliced_prefill have consistent
892	        lengths/non-None status across all DPs.  If any queue is out of
893	        sync, the method returns False so the caller can sleep and retry.
894	
895	        When dp_coord_group is not set or coordination is disabled,
896	        returns True immediately (no sync needed).
897	        """
898	        if self.dp_coord_group is None or not self._is_coordinated_dp():
899	            return True
900	
901	        import torch
902	        import torch.distributed as dist
903	        _dp_size = getattr(
904	            self.vllm_config.parallel_config, "data_parallel_size", 1
905	        )
906	        _dp_rank = getattr(
907	            s
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/6 09:17:35</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github/vllm-ascend/vllm_ascend\" && echo \"=== passive_core.py step() flow (schedule call + idle return) ===\" && grep -n \"def step\\|sync_queue_state\\|poll_and_classify\\|\\.schedule(\\|_schedule_expect_alternation\\|return False\\|return True\\|_idle_sleep\\|has_unfinished\\|_has_global\\|ready_prefills\\|ready_decodes\\|_active\" v1/engine/passive_core.py | head -50",
  "description": "Read step() and schedule structure"
}
```

</details>


---

## User <sup>2026/8/6 09:17:37</sup>

<details>
<summary>Tool Result</summary>

```
=== passive_core.py step() flow (schedule call + idle return) ===
562:        self._idle_sleep_seconds = 0.001
608:    def step(self) -> bool:
627:        self.passive_scheduler.poll_and_classify()
630:        if not self.passive_scheduler.sync_queue_state():
631:            return False
634:        batch = self.passive_scheduler.schedule()
644:            return False
798:        return True
854:                    time.sleep(self._idle_sleep_seconds)
```

</details>


---

## Assistant <sup>2026/8/6 09:17:41</sup>


---

## Assistant <sup>2026/8/6 09:17:41</sup>

让我查看一下 `step()` 的主体，以了解调度是在何处被调用的，以及空闲时的返回路径。


---

## Assistant <sup>2026/8/6 09:17:41</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/v1/engine/passive_core.py",
  "limit": 60,
  "offset": 608
}
```

</details>


---

## User <sup>2026/8/6 09:17:41</sup>

<details>
<summary>Tool Result</summary>

```
608	    def step(self) -> bool:
609	        """Single tick: poll ZMQ → pick batches → enqueue worker payloads.
610	
611	        Batches are dispatched one phase at a time in the order encoded by
612	        the configured dispatch policy.
613	
614	        Cross-DP coordination (when dp>1 + MoE + PD-separation) is handled
615	        internally by :meth:`PassiveScheduler.schedule` via
616	        :meth:`~PassiveScheduler._coordinate_decision`.
617	
618	        Returns:
619	            True if at least one payload was enqueued, False if the
620	            scheduler had nothing to dispatch.
621	        """
622	        _t0 = time.monotonic()
623	        self._drain_worker_completion_acks()
624	        _dt_drain = (time.monotonic() - _t0) * 1000
625	
626	        _t0 = time.monotonic()
627	        self.passive_scheduler.poll_and_classify()
628	        _dt_poll = (time.monotonic() - _t0) * 1000
629	
630	        if not self.passive_scheduler.sync_queue_state():
631	            return False
63
...(truncated)
```

</details>


---

## Assistant <sup>2026/8/6 09:19:33</sup>


---

## Assistant <sup>2026/8/6 09:19:34</sup>

我现在明白了流程。当空闲时，`step()` 仍然会调用 `schedule()`（在 `_schedule_expect_alternation` 中有 2 次 all_reduce），尽管此时并没有工作要做。`sync_queue_state` 已经在各个 DP 之间交换了队列长度——所以在此之后我们就知道是否“所有 DP 都处于空闲状态”。我来查看一下 `sync_queue_state` 的具体返回情况，以实现一个安全的关口。


---

## Assistant <sup>2026/8/6 09:19:35</sup>

<details>
<summary>Tool: <code>Read</code></summary>

```json
{
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py",
  "limit": 14,
  "offset": 934
}
```

</details>


---

## User <sup>2026/8/6 09:19:35</sup>

<details>
<summary>Tool Result</summary>

```
934	        # Verify all rank values are identical for each of the 5 queues.
935	        all_match = True
936	        for q in range(5):
937	            vals = [int(tensor[r * 5 + q].item()) for r in range(_dp_size)]
938	            if len(set(vals)) != 1:
939	                _names = ["ready_prefills", "ready_decodes", "ready_pdmixes",
940	                          "_active_prefill_slices", "_active_sliced_prefill"]
941	                logger.warning(
942	                    "[SYNC-QUEUE] rank=%s %s mismatch: %s",
943	                    _dp_rank, _names[q], vals,
944	                )
945	                all_match = False
946	        return all_match
947
```

</details>


---

## Assistant <sup>2026/8/6 09:20:01</sup>


---

## Assistant <sup>2026/8/6 09:20:03</sup>

修复方案很明确且安全：`sync_queue_state` 已经通过其 all_reduce 交换了所有 DP 间的队列长度，因此在此之后我们就能明确是否“所有 DP 均处于空闲状态”。当全局空闲时，跳过 `schedule()`（即那 2 次 all_reduce）——两个 DP 都会做出相同的决定（相同的数据），因此 barrier 配对得到保留。我将只在有任务时才输出 sync_queue 日志。

让我应用这两处修改。


---

## Assistant <sup>2026/8/6 09:20:06</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py",
  "old_string": "        _cc_t0 = time.monotonic()\n        dist.all_reduce(tensor, op=dist.ReduceOp.SUM,\n                         group=self.dp_coord_group)\n        logger.error(\n            \"[EC-PERF][CLOUD-COORD] dp_rank=%s phase=sync_queue all_reduce=%.3fms\",\n            self.vllm_config.parallel_config.data_parallel_rank,\n            (time.monotonic() - _cc_t0) * 1000,\n        )",
  "new_string": "        _cc_t0 = time.monotonic()\n        dist.all_reduce(tensor, op=dist.ReduceOp.SUM,\n                         group=self.dp_coord_group)\n        _cc_dt_ms = (time.monotonic() - _cc_t0) * 1000\n        # Any work across all cloud DPs? (5 fields/DP: pf/dec/pdmix/slices/active)\n        # After this all_reduce both DPs hold the same view, so both make the\n        # same skip decision -> barrier pairing preserved.\n        has_any_work = any(\n            int(tensor[i].item()) != 0 for i in range(_dp_size * 5)\n        )\n        self._synced_has_any_work = has_any_work\n        # 仅在有工作时打印，避免空闲刷屏（空闲时仍保留 1 次 all_reduce 作为探测）\n        if has_any_work:\n            logger.error(\n                \"[EC-PERF][CLOUD-COORD] dp_rank=%s phase=sync_queue \"\n                \"all_reduce=%.3fms has_work=%s\",\n                self.vllm_config.parallel_config.data_parallel_rank,\n                _cc_dt_ms, has_any_work,\n            )"
}
```

</details>


---

## User <sup>2026/8/6 09:20:07</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/core/passive_scheduler.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/6 09:20:14</sup>


---

## Assistant <sup>2026/8/6 09:20:14</sup>

现在在 `step()` 中添加空闲跳过逻辑：


---

## Assistant <sup>2026/8/6 09:20:16</sup>

<details>
<summary>Tool: <code>Edit</code></summary>

```json
{
  "replace_all": false,
  "file_path": "c:/theroadtofree_github/vllm-ascend/vllm_ascend/v1/engine/passive_core.py",
  "old_string": "        if not self.passive_scheduler.sync_queue_state():\n            return False\n\n        _t0 = time.monotonic()\n        batch = self.passive_scheduler.schedule()\n        _dt_sched = (time.monotonic() - _t0) * 1000",
  "new_string": "        if not self.passive_scheduler.sync_queue_state():\n            return False\n\n        # 空闲跳过调度：sync_queue_state 已跨 DP 交换队列长度，若所有云 DP 均无工作\n        # (ready/active 全空) 则跳过 schedule() 的 2 次 all_reduce + 调度逻辑。\n        # 两云 DP 持相同视图，同跳同不跳，barrier 配对不变；有任一 DP 有工作时\n        # has_any_work=True 照常调度（idle DP 走 replay dummy 配对）。\n        if not getattr(self.passive_scheduler, \"_synced_has_any_work\", True):\n            return False\n\n        _t0 = time.monotonic()\n        batch = self.passive_scheduler.schedule()\n        _dt_sched = (time.monotonic() - _t0) * 1000"
}
```

</details>


---

## User <sup>2026/8/6 09:20:17</sup>

<details>
<summary>Tool Result</summary>

```
The file c:/theroadtofree_github/vllm-ascend/vllm_ascend/v1/engine/passive_core.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


---

## Assistant <sup>2026/8/6 09:20:41</sup>


---

## Assistant <sup>2026/8/6 09:20:42</sup>

我来对这两个修改过的文件进行语法检查。


---

## Assistant <sup>2026/8/6 09:20:42</sup>

<details>
<summary>Tool: <code>Bash</code></summary>

```json
{
  "command": "cd \"c:/theroadtofree_github\" && for f in vllm-ascend/vllm_ascend/core/passive_scheduler.py vllm-ascend/vllm_ascend/v1/engine/passive_core.py; do python -m py_compile \"$f\" 2>&1 && echo \"OK: $f\" || echo \"FAIL: $f\"; done",
  "description": "Syntax-check modified files"
}
```

</details>


---

## User <sup>2026/8/6 09:26:25</sup>

<details>
<summary>Tool Result</summary>

```
OK: vllm-ascend/vllm_ascend/core/passive_scheduler.py
OK: vllm-ascend/vllm_ascend/v1/engine/passive_core.py
```

</details>


---

## Assistant <sup>2026/8/6 09:26:47</sup>


---

## Assistant <sup>2026/8/6 09:26:57</sup>

两处改动语法通过。说明一下这个空闲优化:

## 改动

**1. `passive_scheduler.py` `sync_queue_state`** — all_reduce 交换完各 DP 队列长度后,计算 `has_any_work`(5 个队列字段在任一 DP 上非空即有工作)并存到 `self._synced_has_any_work`;`[EC-PERF][CLOUD-COORD] phase=sync_queue` 日志改为**仅 has_work 时打印**。

**2. `passive_core.py` `step()`** — `sync_queue_state` 之后,若 `_synced_has_any_work` 为 False(所有云 DP 均空闲),直接 `return False`,**跳过 `schedule()`**(即跳过 988/1023 那两次 intent_max / deltas_sum all_reduce + 调度逻辑)。

## 为什么安全(不破坏 barrier 配对)

- `sync_queue_state` 仍**每步都做**(保留 1 次 all_reduce 作为探测):两云 DP 始终同时进入,barrier 配对不变。
- 两云 DP 拿到的是同一份交换后的队列长度,**同跳同不跳**,不会一个进 schedule 一个不进而死锁。
- **任一 DP 有工作**时 `has_any_work=True`,两 DP 照常进 schedule——idle DP 走 `_replay_by_deltas` 跑 dummy-middle 配对。这正是探测不能省的原因:DP0 来 real、DP1 空闲时,DP1 必须靠这次跨 DP 同步才知道该跑 dummy。

## 效果

| 场景 | 改前(空闲) | 改后(空闲) | 有工作时 |
|---|---|---|---|
| all_reduce 次数/步 | 3 | **1**(探测) | 3(不变) |
| 日志行(928/988/1023) | 3 行/步 | **0 行** | 3 行(不变) |

空闲时从 3 次 all_reduce 降到 1 次,且不再刷屏。保留的这 1 次是**探测**:用来发现"对端来活了"并唤醒本侧跑 dummy。若彻底去掉它,DP0 来活、DP1 在 sleep,DP0 会在 schedule 的 all_reduce 上空等 DP1->超时/死锁(边侧虽 dual-send real+dummy ZMQ 给两云,但 ZMQ 到达有 skew,探测正是兜底这个 skew)。

## 进一步可调(可选)

若想再压低空闲开销,可把空闲 sleep 拉长:[passive_core.py:562](vllm-ascend/vllm_ascend/v1/engine/passive_core.py#L562) `self._idle_sleep_seconds = 0.001`(1ms)。比如空闲时用 5ms,探测 all_reduce 频率降 5 倍,代价是最多 5ms 的"来活感知"延迟。要不要顺手改成空闲自适应(连续若干步空闲就拉长 sleep、来活立即归位)我可以再加。


---
