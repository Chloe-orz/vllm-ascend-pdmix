#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

"""Edge-cloud model adapter for the DeepSeek-V4 MTP drafter.

The adapter keeps the public edge-cloud draft contract aligned with Qwen-MTP:

* segment A runs lightweight input preparation on the edge;
* segment C runs the selected MTP decoder block on the cloud;
* segment E returns the pre-``hc_head`` state to the edge.

The existing MTP draft flow then calls ``compute_logits`` on the edge. For
DeepSeek-V4, that method already performs ``hc_head`` and the shared vocabulary
head, while the same pre-``hc_head`` state can be reused by the next draft step.
"""

from collections.abc import Iterable
from typing import Any

import torch
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

import vllm_ascend.models.deepseek_v4_mtp as deepseek_v4_mtp_module
from vllm_ascend.distributed.parallel_state import is_edge_device
from vllm_ascend.models.deepseek_v4_mtp import (
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
    DeepSeekV4MTP,
)

_ORIGINAL_MTP_LAYER_INIT = DeepSeekMultiTokenPredictorLayer.__init__
_ORIGINAL_SET_MOE_PARAMETERS = DeepSeekV4MTP.set_moe_parameters
_ORIGINAL_LOAD_WEIGHTS = DeepSeekV4MTP.load_weights


def _make_missing_decoder_layer(
    *unused_args: Any,
    **unused_kwargs: Any,
) -> PPMissingLayer:
    return PPMissingLayer()


def _deepseek_v4_mtp_layer_init_edge_cloud(
    self: DeepSeekMultiTokenPredictorLayer,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Avoid constructing the cloud-only decoder block on the edge."""
    if not is_edge_device():
        _ORIGINAL_MTP_LAYER_INIT(self, *args, **kwargs)
        return

    original_decoder_layer = deepseek_v4_mtp_module.DeepseekV2DecoderLayer
    deepseek_v4_mtp_module.DeepseekV2DecoderLayer = (
        _make_missing_decoder_layer
    )
    try:
        _ORIGINAL_MTP_LAYER_INIT(self, *args, **kwargs)
    finally:
        deepseek_v4_mtp_module.DeepseekV2DecoderLayer = original_decoder_layer


def _deepseek_v4_mtp_set_moe_parameters_edge_cloud(
    self: DeepSeekV4MTP,
) -> None:
    layers = self.model.layers.values()
    if is_edge_device() and all(
        isinstance(layer, PPMissingLayer)
        or isinstance(layer.mtp_block, PPMissingLayer)
        for layer in layers
    ):
        self.expert_weights = []
        self.moe_layers = []
        self.moe_mlp_layers = []
        self.extract_moe_parameters(None)
        return
    _ORIGINAL_SET_MOE_PARAMETERS(self)


def _deepseek_v4_mtp_load_weights_edge_cloud(
    self: DeepSeekV4MTP,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    if is_edge_device():
        # The edge owns embedding/projection/norm/hc_head/shared-head weights,
        # but its decoder block is a construction-time PPMissingLayer.
        weights = (
            (name, weight)
            for name, weight in weights
            if self.no_mtp_block_in_name(name)
        )
    return _ORIGINAL_LOAD_WEIGHTS(self, weights)


def _get_mtp_layer(
    predictor: DeepSeekMultiTokenPredictor,
    spec_step_idx: int,
) -> DeepSeekMultiTokenPredictorLayer:
    layer_keys = list(predictor.layers.keys())
    if not layer_keys:
        raise RuntimeError("DeepSeek-V4 MTP has no predictor layers")

    layer_key = layer_keys[spec_step_idx % len(layer_keys)]
    layer = predictor.layers[layer_key]
    if isinstance(layer, PPMissingLayer):
        raise RuntimeError(
            "DeepSeek-V4 MTP selected a missing predictor layer: "
            f"key={layer_key}, spec_step_idx={spec_step_idx}"
        )
    if not isinstance(layer, DeepSeekMultiTokenPredictorLayer):
        raise TypeError(
            "Unexpected DeepSeek-V4 MTP predictor layer type: "
            f"{type(layer).__name__}"
        )
    return layer


def _deepseek_v4_mtp_predictor_forward_edge_cloud_segment(
    self: DeepSeekMultiTokenPredictor,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    previous_hidden_states: torch.Tensor | None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
    is_first_segment: bool | None = None,
    is_last_segment: bool | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    del start_layer, end_layer, extra_layer_kwargs

    if is_first_segment is None or is_last_segment is None:
        raise ValueError(
            "DeepSeek-V4 MTP edge-cloud segments require explicit "
            "is_first_segment/is_last_segment"
        )
    if is_first_segment and is_last_segment:
        raise ValueError(
            "DeepSeek-V4 MTP edge-cloud segment cannot be both first and last"
        )

    if is_first_segment:
        layer = _get_mtp_layer(self, spec_step_idx)
        if previous_hidden_states is None:
            raise ValueError(
                "DeepSeek-V4 MTP segment A requires previous_hidden_states"
            )
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "DeepSeek-V4 MTP segment A requires input_ids or "
                    "inputs_embeds"
                )
            inputs_embeds = self.embed_input_ids(input_ids)

        num_tokens = positions.shape[-1]
        hidden_size = layer.config.hidden_size
        if inputs_embeds.shape != (num_tokens, hidden_size):
            raise RuntimeError(
                "DeepSeek-V4 MTP segment A embedding shape mismatch: "
                f"expected={(num_tokens, hidden_size)}, "
                f"got={tuple(inputs_embeds.shape)}"
            )
        expected_hidden_elements = (
            num_tokens * layer.hc_mult * hidden_size
        )
        if previous_hidden_states.numel() != expected_hidden_elements:
            raise RuntimeError(
                "DeepSeek-V4 MTP segment A target hidden-state size "
                "mismatch: "
                f"expected_elements={expected_hidden_elements}, "
                f"got={previous_hidden_states.numel()}"
            )
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0,
            0,
            inputs_embeds,
        )
        inputs_embeds = layer.enorm(inputs_embeds)
        previous_hidden_states = previous_hidden_states.view(
            -1,
            layer.hc_mult,
            layer.config.hidden_size,
        )
        previous_hidden_states = layer.hnorm(previous_hidden_states)
        hidden_states = (
            layer.e_proj(inputs_embeds).unsqueeze(-2)
            + layer.h_proj(previous_hidden_states)
        )
        return IntermediateTensors({"hidden_states": hidden_states})

    if intermediate_tensors is None:
        raise ValueError(
            "DeepSeek-V4 MTP segment C/E requires intermediate_tensors"
        )
    hidden_states = intermediate_tensors["hidden_states"]
    num_tokens = positions.shape[-1]
    if hidden_states.ndim != 3 or hidden_states.shape[0] != num_tokens:
        raise RuntimeError(
            "DeepSeek-V4 MTP segment C/E hidden-state shape mismatch: "
            f"tokens={num_tokens}, shape={tuple(hidden_states.shape)}"
        )

    if is_last_segment:
        # Keep the HC state uncollapsed. The shared MTP draft flow invokes
        # DeepSeekV4MTP.compute_logits on the edge, which applies hc_head and
        # leaves this tensor available as the next draft step's hidden state.
        return hidden_states

    layer = _get_mtp_layer(self, spec_step_idx)
    expected_hidden_shape = (
        num_tokens,
        layer.hc_mult,
        layer.config.hidden_size,
    )
    if tuple(hidden_states.shape) != expected_hidden_shape:
        raise RuntimeError(
            "DeepSeek-V4 MTP segment C hidden-state shape mismatch: "
            f"expected={expected_hidden_shape}, "
            f"got={tuple(hidden_states.shape)}"
        )
    if isinstance(layer.mtp_block, PPMissingLayer):
        raise RuntimeError(
            "DeepSeek-V4 MTP segment C cannot execute a missing mtp_block"
        )
    hidden_states, residual = layer.mtp_block(
        positions=positions,
        hidden_states=hidden_states,
        residual=None,
    )
    return IntermediateTensors(
        {
            "hidden_states": hidden_states,
            "residual": residual,
        }
    )


def _deepseek_v4_mtp_forward_edge_cloud_segment(
    self: DeepSeekV4MTP,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    previous_hidden_states = extra_layer_kwargs.pop("hidden_states", None)
    spec_step_idx = extra_layer_kwargs.pop("spec_step_idx", 0)
    return self.model.forward_edge_cloud_segment(
        start_layer,
        end_layer,
        input_ids,
        positions,
        previous_hidden_states,
        intermediate_tensors,
        inputs_embeds,
        spec_step_idx,
        **extra_layer_kwargs,
    )


def _deepseek_v4_mtp_make_empty_intermediate_tensors(
    self: DeepSeekMultiTokenPredictor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    layer = _get_mtp_layer(self, 0)
    tensor_shape = (
        batch_size,
        layer.hc_mult,
        layer.config.hidden_size,
    )
    return IntermediateTensors(
        {
            "hidden_states": torch.zeros(
                tensor_shape,
                dtype=dtype,
                device=device,
            ),
            "residual": torch.zeros(
                tensor_shape,
                dtype=dtype,
                device=device,
            ),
        }
    )


def _deepseek_v4_mtp_model_make_empty_intermediate_tensors(
    self: DeepSeekV4MTP,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    return self.model.make_empty_intermediate_tensors(
        batch_size,
        dtype,
        device,
    )


def _replace_module_with_missing(
    module: torch.nn.Module,
    module_name: str,
) -> None:
    child = getattr(module, module_name, None)
    if child is not None and not isinstance(child, PPMissingLayer):
        setattr(module, module_name, PPMissingLayer())


def _validate_module_ownership(
    module: torch.nn.Module,
    module_name: str,
    *,
    should_be_missing: bool,
    role: str,
) -> None:
    child = getattr(module, module_name, None)
    if child is None:
        raise RuntimeError(
            "DeepSeek-V4 MTP edge-cloud sharding cannot find "
            f"{module_name!r} on {type(module).__name__}"
        )
    is_missing = isinstance(child, PPMissingLayer)
    if is_missing != should_be_missing:
        expected = "PPMissingLayer" if should_be_missing else "a real module"
        raise RuntimeError(
            "DeepSeek-V4 MTP edge-cloud sharding assigned an invalid "
            f"{module_name!r} on {role}: expected {expected}, "
            f"got {type(child).__name__}"
        )


def _validate_deepseek_v4_mtp_shard(
    model: DeepSeekV4MTP,
    *,
    is_edge: bool,
) -> None:
    role = "edge" if is_edge else "cloud"
    predictor = model.model
    _validate_module_ownership(
        predictor,
        "embed_tokens",
        should_be_missing=not is_edge,
        role=role,
    )
    for layer in predictor.layers.values():
        if not isinstance(layer, DeepSeekMultiTokenPredictorLayer):
            raise RuntimeError(
                "DeepSeek-V4 MTP edge-cloud sharding requires real predictor "
                f"layers on {role}, got {type(layer).__name__}"
            )
        _validate_module_ownership(
            layer,
            "mtp_block",
            should_be_missing=is_edge,
            role=role,
        )
        for module_name in (
            "e_proj",
            "h_proj",
            "enorm",
            "hnorm",
            "shared_head",
        ):
            _validate_module_ownership(
                layer,
                module_name,
                should_be_missing=not is_edge,
                role=role,
            )


def _deepseek_v4_mtp_shard_for_edge_cloud(
    self: DeepSeekV4MTP,
    *,
    is_edge: bool,
) -> None:
    predictor = self.model
    if is_edge:
        for layer in predictor.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepSeekMultiTokenPredictorLayer):
                raise TypeError(
                    "Unexpected DeepSeek-V4 MTP predictor layer type: "
                    f"{type(layer).__name__}"
                )
            _replace_module_with_missing(layer, "mtp_block")

        # The edge owns no MTP MoE blocks after sharding. Clear the metadata
        # without calling set_moe_parameters(), whose stock implementation
        # expects every layer to contain a real decoder block.
        self.expert_weights = []
        self.moe_layers = []
        self.moe_mlp_layers = []
        self.extract_moe_parameters(None)
        _validate_deepseek_v4_mtp_shard(self, is_edge=True)
        return

    _replace_module_with_missing(predictor, "embed_tokens")
    for layer in predictor.layers.values():
        if isinstance(layer, PPMissingLayer):
            continue
        if not isinstance(layer, DeepSeekMultiTokenPredictorLayer):
            raise TypeError(
                "Unexpected DeepSeek-V4 MTP predictor layer type: "
                f"{type(layer).__name__}"
            )
        for module_name in (
            "e_proj",
            "h_proj",
            "enorm",
            "hnorm",
            "shared_head",
        ):
            _replace_module_with_missing(layer, module_name)

    # The cloud retains the real MTP blocks, so the stock collector remains
    # valid and keeps EPLB/EP metadata aligned with the sharded module tree.
    self.set_moe_parameters()
    _validate_deepseek_v4_mtp_shard(self, is_edge=False)


DeepSeekMultiTokenPredictor.forward_edge_cloud_segment = (
    _deepseek_v4_mtp_predictor_forward_edge_cloud_segment
)
DeepSeekMultiTokenPredictor.make_empty_intermediate_tensors = (
    _deepseek_v4_mtp_make_empty_intermediate_tensors
)
DeepSeekV4MTP.forward_edge_cloud_segment = (
    _deepseek_v4_mtp_forward_edge_cloud_segment
)
DeepSeekV4MTP.make_empty_intermediate_tensors = (
    _deepseek_v4_mtp_model_make_empty_intermediate_tensors
)
DeepSeekMultiTokenPredictorLayer.__init__ = (
    _deepseek_v4_mtp_layer_init_edge_cloud
)
DeepSeekV4MTP.set_moe_parameters = (
    _deepseek_v4_mtp_set_moe_parameters_edge_cloud
)
DeepSeekV4MTP.load_weights = _deepseek_v4_mtp_load_weights_edge_cloud
DeepSeekV4MTP.shard_for_edge_cloud = _deepseek_v4_mtp_shard_for_edge_cloud
DeepSeekV4MTP.edge_cloud_draft_kind = "deepseek_v4_mtp"
DeepSeekV4MTP.edge_cloud_dynamic_step_segments = True
DeepSeekV4MTP.edge_cloud_uses_dsa_draft_metadata = True
DeepSeekV4MTP.edge_cloud_attention_on_cloud_only = True
