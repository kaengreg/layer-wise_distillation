"""Block Influence calculation and deterministic layer ranking."""

from collections.abc import Iterable, Sequence
import math

import torch
import torch.nn.functional as F


def block_influence_from_hidden_states(hidden_states: Sequence[torch.Tensor], attention_mask: torch.Tensor) -> list[float]:
    """Return one sequence-averaged Block Influence score per decoder block."""
    if len(hidden_states) < 2:
        raise ValueError("hidden_states must contain the embedding output and at least one block output")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")

    mask = attention_mask.bool()
    scores = []
    for block_input, block_output in zip(hidden_states[:-1], hidden_states[1:]):
        cosine = F.cosine_similarity(block_input.float(), block_output.float(), dim=-1)
        valid_per_sequence = mask.sum(dim=1).clamp_min(1)
        sequence_scores = 1.0 - (cosine * mask).sum(dim=1) / valid_per_sequence
        nonempty = mask.any(dim=1)
        if not nonempty.any():
            raise ValueError("attention_mask contains no non-padding tokens")
        scores.append(sequence_scores[nonempty].mean().item())
    return scores


def block_influence_from_activations(block_inputs: Sequence[torch.Tensor], block_outputs: Sequence[torch.Tensor], attention_mask: torch.Tensor) -> list[float]:
    """Return Block Influence from exact decoder-block inputs and outputs."""
    if not block_inputs or len(block_inputs) != len(block_outputs):
        raise ValueError("block input and output activations must have the same non-zero length")
    interleaved = []
    for block_input, block_output in zip(block_inputs, block_outputs):
        interleaved.extend((block_input, block_output))
    scores = []
    for index in range(0, len(interleaved), 2):
        scores.append(block_influence_from_hidden_states(interleaved[index : index + 2], attention_mask)[0])
    return scores


def accumulate_block_influence(score_sums: list[float] | None, hidden_states: Sequence[torch.Tensor], attention_mask: torch.Tensor) -> tuple[list[float], int]:
    """Accumulate per-sequence scores for a batch without weighting longer sequences more."""
    mask = attention_mask.bool()
    nonempty = mask.any(dim=1)
    if not nonempty.any():
        return score_sums or [0.0] * (len(hidden_states) - 1), 0

    batch_sums = []
    for block_input, block_output in zip(hidden_states[:-1], hidden_states[1:]):
        cosine = F.cosine_similarity(block_input.float(), block_output.float(), dim=-1)
        valid_per_sequence = mask.sum(dim=1).clamp_min(1)
        influence = 1.0 - (cosine * mask).sum(dim=1) / valid_per_sequence
        batch_sums.append(influence[nonempty].sum().item())
    if score_sums is None:
        score_sums = [0.0] * len(batch_sums)
    return [total + value for total, value in zip(score_sums, batch_sums)], int(nonempty.sum().item())


def rank_layers(scores: Sequence[float], num_layers: int, protect_first: int = 0, protect_last: int = 0) -> list[int]:
    """Rank eligible layers from least to most influential with stable index ties."""
    if len(scores) != num_layers:
        raise ValueError(f"expected {num_layers} scores, received {len(scores)}")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("Block Influence scores must all be finite")
    if protect_first < 0 or protect_last < 0 or protect_first + protect_last >= num_layers:
        raise ValueError("protected ranges must leave at least one eligible layer")
    eligible = range(protect_first, num_layers - protect_last)
    return sorted(eligible, key=lambda layer_idx: (float(scores[layer_idx]), layer_idx))


@torch.no_grad()
def evaluate_block_influence(model: torch.nn.Module, batches: Iterable[dict[str, torch.Tensor]], device: torch.device) -> list[float]:
    """Evaluate Block Influence over tokenized batches."""
    from .pruning import decoder_layers

    model.eval()
    totals = None
    sequence_count = 0
    for batch in batches:
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        block_inputs = []
        block_outputs = []

        def capture_activations(module, inputs, output):
            block_inputs.append(inputs[0])
            block_outputs.append(output[0] if isinstance(output, tuple) else output)

        handles = [layer.register_forward_hook(capture_activations) for layer in decoder_layers(model)]
        try:
            model(**batch, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        batch_scores = block_influence_from_activations(block_inputs, block_outputs, batch["attention_mask"])
        count = int(batch["attention_mask"].bool().any(dim=1).sum().item())
        weighted_scores = [score * count for score in batch_scores]
        totals = weighted_scores if totals is None else [total + score for total, score in zip(totals, weighted_scores)]
        sequence_count += count
    if not sequence_count:
        raise ValueError("importance data contains no non-padding sequences")
    return [score / sequence_count for score in totals]
