import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerLayerwiseAuxOutput(unittest.TestCase):
    def test_preserves_intermediate_tensors(self):
        intermediate = IntermediateTensors(
            {
                "hidden_states": torch.randn(2, 4),
                "residual": torch.randn(2, 4),
                "input_embeds": torch.randn(2, 4),
            }
        )

        result = NPUModelRunner._unwrap_layerwise_aux_hidden_state_output(
            intermediate
        )

        self.assertIs(result, intermediate)

    def test_unwraps_final_aux_hidden_state_pair(self):
        hidden_states = torch.randn(2, 4)
        aux_hidden_states = [torch.randn(2, 4)]

        result = NPUModelRunner._unwrap_layerwise_aux_hidden_state_output(
            (hidden_states, aux_hidden_states)
        )

        self.assertIs(result, hidden_states)

    def test_accumulates_eagle3_aux_states_across_layer_slices(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner._eagle3_cloud_aux_hidden_states = None
        runner._eagle3_cloud_aux_hidden_states_by_task = {}
        runner._uses_scheduled_edge_cloud_draft = MagicMock(return_value=True)
        runner.speculative_config = SimpleNamespace(method="eagle3")
        runner._last_scheduler_output = SimpleNamespace(
            head_token="interleaved-decode"
        )
        runner._layerwise_scheduler_output = SimpleNamespace(
            head_token="sliced-target"
        )
        first_slice = SimpleNamespace(is_first_slice=True)
        middle_slice = SimpleNamespace(is_first_slice=False)
        first_aux = torch.randn(2, 4)
        last_aux = torch.randn(2, 8)

        runner._cache_eagle3_cloud_aux_hidden_states(
            first_aux, first_slice
        )
        # Slices that contain no configured Eagle3 layer must preserve the
        # parts collected by earlier slices.
        runner._cache_eagle3_cloud_aux_hidden_states(None, middle_slice)
        runner._cache_eagle3_cloud_aux_hidden_states(
            last_aux, middle_slice
        )

        parts = runner._eagle3_cloud_aux_hidden_states_by_task[
            "sliced-target"
        ]
        self.assertIsInstance(parts, list)
        self.assertIs(parts[0], first_aux)
        self.assertIs(parts[1], last_aux)
        actual = NPUModelRunner._combine_eagle3_cloud_aux_hidden_states(
            parts
        )
        expected = torch.cat((first_aux, last_aux), dim=-1)
        torch.testing.assert_close(actual, expected)
        self.assertNotIn(
            "interleaved-decode",
            runner._eagle3_cloud_aux_hidden_states_by_task,
        )

    def test_first_layer_slice_resets_eagle3_aux_accumulator(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner._eagle3_cloud_aux_hidden_states = None
        runner._eagle3_cloud_aux_hidden_states_by_task = {
            "sliced-target": [torch.randn(2, 12)]
        }
        runner._uses_scheduled_edge_cloud_draft = MagicMock(return_value=True)
        runner.speculative_config = SimpleNamespace(method="eagle3")
        runner._last_scheduler_output = None
        runner._layerwise_scheduler_output = SimpleNamespace(
            head_token="sliced-target"
        )
        new_aux = torch.randn(2, 4)

        runner._cache_eagle3_cloud_aux_hidden_states(
            new_aux, SimpleNamespace(is_first_slice=True)
        )

        parts = runner._eagle3_cloud_aux_hidden_states_by_task[
            "sliced-target"
        ]
        self.assertEqual(len(parts), 1)
        self.assertIs(parts[0], new_aux)

    def test_rejects_mismatched_eagle3_aux_slice_shapes(self):
        with self.assertRaisesRegex(
            RuntimeError, "incompatible auxiliary hidden-state shapes"
        ):
            NPUModelRunner._combine_eagle3_cloud_aux_hidden_states(
                [torch.randn(2, 4), torch.randn(3, 4)]
            )

    def test_unsliced_eagle3_aux_state_is_cloned_for_graph_replay(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner._eagle3_cloud_aux_hidden_states = None
        runner._eagle3_cloud_aux_hidden_states_by_task = {}
        runner._uses_scheduled_edge_cloud_draft = MagicMock(return_value=True)
        runner.speculative_config = SimpleNamespace(method="eagle3")
        runner._last_scheduler_output = SimpleNamespace(
            head_token="unsliced-target"
        )
        aux_hidden_states = torch.randn(2, 12)

        runner._cache_eagle3_cloud_aux_hidden_states(
            aux_hidden_states, None
        )

        cached = runner._eagle3_cloud_aux_hidden_states_by_task[
            "unsliced-target"
        ]
        self.assertIsInstance(cached, torch.Tensor)
        self.assertIsNot(cached, aux_hidden_states)
        torch.testing.assert_close(cached, aux_hidden_states)


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_compress = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestNPUModelRunnerOutputTokenIds(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.model_config = MagicMock()
        runner.use_compress = False
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_output_token_ids_before_sampler(self, mock_lmhead_tp_enable):
        """Verify output_token_ids are updated before sampler is called"""
        mock_lmhead_tp_enable.return_value = False

        # Build input batch with historical sampled tokens
        input_batch = MagicMock()
        input_batch.sampling_metadata.output_token_ids = [
            [1, 2, 3, -1],
            [4, 5, -1],
        ]
        input_batch.num_reqs = 2
        input_batch.top_k_cpu = None
        input_batch.prev_req_id_to_index = {
            "req0": 0,
            "req1": 1,
        }
        input_batch.sampled_token_ids_cpu = torch.tensor([6, 7])
        input_batch.async_copy_ready_event = MagicMock()
        input_batch.async_copy_ready_event.synchronize = MagicMock()

        # Simulate the real behavior of InputBatch.update_async_output_token_ids
        def mock_update_output_token_ids():
            output_token_ids = input_batch.sampling_metadata.output_token_ids
            sampled_ids = input_batch.sampled_token_ids_cpu.tolist()

            for index, req_id in enumerate(input_batch.prev_req_id_to_index):
                prev_index = input_batch.prev_req_id_to_index[req_id]
                req_output = output_token_ids[index]
                if req_output and req_output[-1] == -1:
                    req_output[-1] = sampled_ids[prev_index]

        input_batch.update_async_output_token_ids.side_effect = mock_update_output_token_ids

        # Build runner and inject dependencies
        runner = self._build_runner()
        runner.input_batch = input_batch
        runner.sampler = MagicMock(return_value=MagicMock())

        # Call sample method
        logits = torch.randn(2, 32000)
        runner._sample(logits=logits, spec_decode_metadata=None)

        # Verify sampler and update_async_output_token_ids were called
        runner.sampler.assert_called_once()
        input_batch.update_async_output_token_ids.assert_called_once()

        # Verify output_token_ids were updated before sampler is called
        call_kwargs = runner.sampler.call_args[1]
        actual_sampling_metadata = call_kwargs["sampling_metadata"]
        actual_output_token_ids = actual_sampling_metadata.output_token_ids
        self.assertEqual(actual_output_token_ids[0], [1, 2, 3, 6])
        self.assertEqual(actual_output_token_ids[1], [4, 5, 7])


class TestNPUModelRunnerDebugger(unittest.TestCase):
    def _build_runner(self, debugger=None):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.debugger = debugger or MagicMock()
        runner.model = MagicMock()
        runner.model_config = MagicMock()
        runner.model_config.enforce_eager = False
        runner._debugger_started = True
        runner._debugger_step_dummy_data_before_execute = False
        runner.use_compress = False
        return runner

    def test_finalize_dump_data_stops_stop_capable_debugger(self):
        runner = self._build_runner()

        runner._finalize_dump_data()

        runner.debugger.stop.assert_called_once_with()
        runner.debugger.step.assert_called_once_with()
        self.assertFalse(runner._debugger_started)

    def test_finalize_dump_data_steps_graph_debugger_without_stop(self):
        debugger = MagicMock(spec=["start", "step"])
        runner = self._build_runner(debugger)

        runner._finalize_dump_data()

        debugger.step.assert_called_once_with()
        self.assertTrue(runner._debugger_started)

    def test_start_dump_data_noop_when_already_started(self):
        runner = self._build_runner(MagicMock(spec=["start", "step"]))

        runner._start_dump_data()

        runner.debugger.start.assert_not_called()
        runner.debugger.step.assert_not_called()
        self.assertTrue(runner._debugger_started)


if __name__ == "__main__":
    unittest.main()
