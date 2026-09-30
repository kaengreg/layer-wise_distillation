"""Repair-layer selection and parameter freezing."""

from collections.abc import Sequence

from torch import nn

from .pruning import decoder_layers


def _contiguous_segments(indices: Sequence[int]) -> list[tuple[int, int]]:
    sorted_indices = sorted(set(indices))
    segments = []
    start = previous = sorted_indices[0]
    for index in sorted_indices[1:]:
        if index != previous + 1:
            segments.append((start, previous))
            start = index
        previous = index
    segments.append((start, previous))
    return segments


def select_repair_layers(old_depth: int, removed_indices: Sequence[int], radius: int = 1) -> list[int]:
    """Return post-pruning indices neighboring every contiguous removed segment."""
    if radius < 0:
        raise ValueError("repair radius cannot be negative")
    removed = sorted(set(removed_indices))
    if not removed or removed[0] < 0 or removed[-1] >= old_depth:
        raise ValueError("removed layer indices are invalid for the pre-pruning depth")
    survivors = [index for index in range(old_depth) if index not in set(removed)]
    old_to_new = {old: new for new, old in enumerate(survivors)}
    selected = set()
    for start, end in _contiguous_segments(removed):
        candidates = range(max(0, start - radius), min(old_depth, end + radius + 1))
        selected.update(old_to_new[index] for index in candidates if index in old_to_new)
    return sorted(selected)


def set_trainable_parameters(model: nn.Module, repair_layers: Sequence[int], train_final_norm: bool = True, train_lm_head: bool = True) -> list[nn.Parameter]:
    """Freeze the student except explicitly selected repair components."""
    layers = decoder_layers(model)
    if any(index < 0 or index >= len(layers) for index in repair_layers):
        raise ValueError("repair layer index is out of range")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for index in repair_layers:
        for parameter in layers[index].parameters():
            parameter.requires_grad = True
    final_norm = getattr(model.model, "norm", None)
    if train_final_norm and final_norm is not None:
        for parameter in final_norm.parameters():
            parameter.requires_grad = True
    lm_head = getattr(model, "lm_head", None)
    if train_lm_head and lm_head is not None:
        for parameter in lm_head.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
