"""In-memory pruning for decoder models exposing ``model.layers``."""

from collections.abc import Sequence

import torch
from torch import nn


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("model must expose decoder layers as model.layers")
    return layers


def _update_config(config, survivors: list[int], old_depth: int) -> None:
    config.num_hidden_layers = len(survivors)
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, (list, tuple)) and len(layer_types) == old_depth:
        config.layer_types = [layer_types[index] for index in survivors]
    max_window_layers = getattr(config, "max_window_layers", None)
    if isinstance(max_window_layers, int):
        config.max_window_layers = sum(index < max_window_layers for index in survivors)


def prune_layers(model: nn.Module, remove_indices: Sequence[int], original_layer_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Remove current student layers and return remaining and removed teacher IDs."""
    layers = decoder_layers(model)
    old_depth = len(layers)
    indices = sorted(set(remove_indices))
    if len(indices) != len(remove_indices):
        raise ValueError("layer indices to remove must be unique")
    if len(original_layer_ids) != old_depth:
        raise ValueError("original_layer_ids must map every current student layer")
    if not indices or indices[0] < 0 or indices[-1] >= old_depth:
        raise ValueError(f"layer indices must be in [0, {old_depth - 1}]")
    if len(indices) == old_depth:
        raise ValueError("cannot remove all decoder layers")

    removed = set(indices)
    survivors = [index for index in range(old_depth) if index not in removed]
    model.model.layers = nn.ModuleList([layers[index] for index in survivors])
    configs = {id(config): config for config in (model.config, model.model.config) if config is not None}
    for config in configs.values():
        _update_config(config, survivors, old_depth)
    for layer_idx, layer in enumerate(model.model.layers):
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "layer_idx"):
            attention.layer_idx = layer_idx

    remaining_original_ids = [int(original_layer_ids[index]) for index in survivors]
    removed_original_ids = [int(original_layer_ids[index]) for index in indices]
    return remaining_original_ids, removed_original_ids
