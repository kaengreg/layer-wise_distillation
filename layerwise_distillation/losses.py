"""Padding-aware distillation losses."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _masked_token_mean(values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(device=values.device, dtype=values.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        raise ValueError("loss mask contains no valid tokens")
    return (values * mask).sum() / denominator


def causal_target_mask(attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return valid next-token transitions with no padding on either side."""
    shift_labels = labels[:, 1:].to(attention_mask.device)
    return attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool() & shift_labels.ne(-100)


def token_kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, attention_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Temperature-scaled forward KL(teacher || student), averaged over valid tokens."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return _masked_token_mean(per_token, attention_mask) * temperature**2


def mapped_block_output_mse(student_block_outputs: Sequence[torch.Tensor], teacher_block_outputs: Sequence[torch.Tensor], original_layer_ids: Sequence[int], attention_mask: torch.Tensor, layer_indices: Sequence[int] | None = None) -> torch.Tensor:
    """Match exact raw decoder-block outputs using original teacher indices."""
    if len(student_block_outputs) != len(original_layer_ids):
        raise ValueError("original_layer_ids must map every student block output")
    selected = list(range(len(original_layer_ids))) if layer_indices is None else list(layer_indices)
    if any(index < 0 or index >= len(student_block_outputs) for index in selected):
        raise ValueError("hidden-state repair layer index is out of range")
    if any(int(teacher_index) < 0 or int(teacher_index) >= len(teacher_block_outputs) for teacher_index in original_layer_ids):
        raise ValueError("original teacher layer mapping is out of range")
    if not selected:
        return student_block_outputs[0].new_zeros(())
    losses = []
    for student_index in selected:
        teacher_index = int(original_layer_ids[student_index])
        squared_error = (student_block_outputs[student_index].float() - teacher_block_outputs[teacher_index].float()).pow(2).mean(dim=-1)
        losses.append(_masked_token_mean(squared_error, attention_mask))
    return torch.stack(losses).mean()


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross entropy excluding padding and ignored labels."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:].to(logits.device)
    valid = causal_target_mask(attention_mask.to(logits.device), labels.to(logits.device))
    if not valid.any():
        raise ValueError("causal language-model loss has no valid target tokens")
    per_token = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels.clamp_min(0), reduction="none")
    return per_token[valid].mean()


def distillation_loss(student_outputs, teacher_outputs, original_layer_ids: Sequence[int], attention_mask: torch.Tensor, labels: torch.Tensor, repair_layers: Sequence[int], temperature: float, kl_weight: float, hidden_weight: float, lm_weight: float, student_block_outputs: Sequence[torch.Tensor] | None = None, teacher_block_outputs: Sequence[torch.Tensor] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine all enabled loss components and return their detached diagnostics."""
    zero = student_outputs.logits.new_zeros(())
    kl = token_kl_divergence(student_outputs.logits, teacher_outputs.logits, attention_mask, temperature) if kl_weight else zero
    if hidden_weight:
        if student_block_outputs is None or teacher_block_outputs is None:
            raise ValueError("hidden-state distillation requires exact decoder-block outputs")
        hidden = mapped_block_output_mse(student_block_outputs, teacher_block_outputs, original_layer_ids, attention_mask, repair_layers)
    else:
        hidden = zero
    lm = causal_lm_loss(student_outputs.logits, labels, attention_mask) if lm_weight else zero
    total = kl_weight * kl + hidden_weight * hidden + lm_weight * lm
    return total, {"loss": total.detach(), "kl_loss": kl.detach(), "hidden_loss": hidden.detach(), "lm_loss": lm.detach()}
