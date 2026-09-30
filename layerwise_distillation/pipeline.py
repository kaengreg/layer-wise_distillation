"""GPU-facing orchestration for iterative pruning and repair distillation."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .importance import evaluate_block_influence, rank_layers
from .losses import causal_lm_loss, causal_target_mask, distillation_loss, token_kl_divergence
from .pruning import decoder_layers, prune_layers
from .repair import select_repair_layers, set_trainable_parameters


@dataclass
class PipelineConfig:
    teacher_model_path: str
    dataset: str
    text_column: str
    target_num_layers: int
    layers_per_iteration: int
    output_dir: str
    dataset_split: str = "train"
    max_importance_samples: int = 512
    max_train_samples: int = 20_000
    max_eval_samples: int = 1_000
    max_sequence_length: int = 512
    protect_first_layers: int = 1
    protect_last_layers: int = 1
    repair_radius: int = 1
    temperature: float = 2.0
    kl_weight: float = 1.0
    hidden_weight: float = 1.0
    lm_weight: float = 0.0
    max_train_steps: int = 200
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    train_final_norm: bool = True
    train_lm_head: bool = True
    dtype: str = "bfloat16"
    attention_implementation: str | None = "sdpa"
    seed: int = 1337
    report_to: str = "none"


def validate_pipeline_config(config: PipelineConfig, initial_num_layers: int | None = None) -> None:
    """Validate all cheap user-facing errors before model weights are loaded."""
    positive_fields = {
        "target_num_layers": config.target_num_layers,
        "layers_per_iteration": config.layers_per_iteration,
        "max_importance_samples": config.max_importance_samples,
        "max_train_samples": config.max_train_samples,
        "max_eval_samples": config.max_eval_samples,
        "max_sequence_length": config.max_sequence_length,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
    }
    invalid = [name for name, value in positive_fields.items() if value <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {', '.join(invalid)}")
    if config.max_train_steps < 0:
        raise ValueError("max_train_steps cannot be negative")
    if config.protect_first_layers < 0 or config.protect_last_layers < 0 or config.repair_radius < 0:
        raise ValueError("protected layer counts and repair radius cannot be negative")
    if config.temperature <= 0 or config.learning_rate <= 0:
        raise ValueError("temperature and learning rate must be positive")
    if min(config.kl_weight, config.hidden_weight, config.lm_weight) < 0:
        raise ValueError("loss weights cannot be negative")
    if config.max_train_steps and config.kl_weight + config.hidden_weight + config.lm_weight == 0:
        raise ValueError("at least one loss weight must be non-zero when training is enabled")
    if config.max_train_steps and config.repair_radius == 0 and not config.train_final_norm and not config.train_lm_head:
        raise ValueError("repair_radius=0 selects no trainable layers when final norm and LM head training are disabled")
    if config.max_train_steps and config.repair_radius == 0 and config.hidden_weight and not config.kl_weight and not config.lm_weight:
        raise ValueError("hidden-state loss requires at least one repair layer because it does not update final norm or LM head")
    if config.dtype == "float16" and not torch.cuda.is_available():
        raise ValueError("float16 execution requires CUDA")
    if initial_num_layers is not None:
        protected = config.protect_first_layers + config.protect_last_layers
        if config.target_num_layers >= initial_num_layers:
            raise ValueError(f"target_num_layers must be smaller than the teacher depth ({initial_num_layers})")
        if config.target_num_layers < max(1, protected):
            raise ValueError("target_num_layers cannot be smaller than the protected layer count")


def validate_single_process_environment(environment: dict[str, str] | None = None) -> None:
    """Reject distributed launches that would race on one model and output directory."""
    environment = os.environ if environment is None else environment
    world_size = int(environment.get("WORLD_SIZE", "1"))
    slurm_tasks = int(environment.get("SLURM_NTASKS", "1"))
    local_rank = int(environment.get("LOCAL_RANK", "0"))
    if world_size > 1 or slurm_tasks > 1 or local_rank != 0:
        raise ValueError("this pipeline supports one process on one GPU; do not launch it with torchrun or multi-task srun")


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output_dir must be empty to avoid mixing stale iteration artifacts: {path}")
    path.mkdir(parents=True, exist_ok=True)


class TextBatcher:
    """Re-iterable tokenizer-backed batches over a fixed list of texts."""

    def __init__(self, texts: list[str], tokenizer, batch_size: int, max_length: int, shuffle: bool = False, seed: int = 0):
        self.texts = texts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(self.texts, batch_size=self.batch_size, shuffle=self.shuffle, generator=generator)
        for texts in loader:
            yield self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")


def _load_text_splits(config: PipelineConfig) -> tuple[list[str], list[str], list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(config.dataset, split=config.dataset_split)
    if config.text_column not in dataset.column_names:
        raise ValueError(f"text column {config.text_column!r} is absent; available columns: {dataset.column_names}")
    required = config.max_train_samples + config.max_eval_samples
    dataset = dataset.shuffle(seed=config.seed).select(range(min(required, len(dataset))))
    texts = [str(text) for text in dataset[config.text_column] if text is not None and str(text).strip()]
    if len(texts) < 2:
        raise ValueError("dataset must contain at least two non-empty texts")
    eval_count = min(config.max_eval_samples, max(1, len(texts) // 10))
    eval_texts = texts[:eval_count]
    train_texts = texts[eval_count : eval_count + config.max_train_samples]
    if not train_texts:
        raise ValueError("no training texts remain after creating the evaluation split")
    importance_texts = train_texts[: config.max_importance_samples]
    return importance_texts, train_texts, eval_texts


def _freeze_teacher(teacher: torch.nn.Module) -> None:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def _forward_with_block_outputs(model, batch: dict[str, torch.Tensor]):
    block_outputs = []

    def capture_output(module, inputs, output):
        block_outputs.append(output[0] if isinstance(output, tuple) else output)

    handles = [layer.register_forward_hook(capture_output) for layer in decoder_layers(model)]
    try:
        outputs = model(**batch, output_hidden_states=False, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return outputs, block_outputs


def _train_iteration(teacher, student, batches, original_layer_ids: list[int], repair_layers: list[int], config: PipelineConfig, device: torch.device) -> dict[str, float]:
    trainable = set_trainable_parameters(student, repair_layers, config.train_final_norm, config.train_lm_head)
    if config.max_train_steps == 0:
        return {"steps": 0, "loss": 0.0, "kl_loss": 0.0, "hidden_loss": 0.0, "lm_loss": 0.0}
    if not trainable:
        raise ValueError("no student parameters were selected for repair")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "kl_loss": 0.0, "hidden_loss": 0.0, "lm_loss": 0.0}
    optimizer_steps = 0
    micro_step = 0
    while optimizer_steps < config.max_train_steps:
        for batch in batches:
            batch = _move_batch(batch, device)
            with torch.no_grad():
                teacher_outputs, teacher_block_outputs = _forward_with_block_outputs(teacher, batch)
            student_outputs, student_block_outputs = _forward_with_block_outputs(student, batch)
            loss, metrics = distillation_loss(student_outputs, teacher_outputs, original_layer_ids, batch["attention_mask"], batch["input_ids"], repair_layers, config.temperature, config.kl_weight, config.hidden_weight, config.lm_weight, student_block_outputs, teacher_block_outputs)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite distillation loss at optimizer step {optimizer_steps + 1}")
            (loss / config.gradient_accumulation_steps).backward()
            micro_step += 1
            for name, value in metrics.items():
                totals[name] += float(value.item())
            if micro_step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=float("inf"), error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps >= config.max_train_steps:
                    break
    denominator = max(1, micro_step)
    return {"steps": optimizer_steps, **{name: value / denominator for name, value in totals.items()}}


@torch.no_grad()
def _evaluate(teacher, student, batches, config: PipelineConfig, device: torch.device) -> dict[str, Any]:
    teacher.eval()
    student.eval()
    lm_total = kl_total = 0.0
    token_count = target_count = 0
    inference_seconds = 0.0
    for batch in batches:
        batch = _move_batch(batch, device)
        teacher_outputs = teacher(**batch, use_cache=False, return_dict=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        student_outputs = student(**batch, use_cache=False, return_dict=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started
        valid_targets = int(causal_target_mask(batch["attention_mask"], batch["input_ids"]).sum().item())
        target_count += valid_targets
        if valid_targets:
            lm_total += float(causal_lm_loss(student_outputs.logits, batch["input_ids"], batch["attention_mask"]).item()) * valid_targets
        kl_total += float(token_kl_divergence(student_outputs.logits, teacher_outputs.logits, batch["attention_mask"], config.temperature).item()) * int(batch["attention_mask"].sum().item())
        token_count += int(batch["attention_mask"].sum().item())
    if not target_count:
        raise ValueError("evaluation data contains no valid next-token targets")
    validation_loss = lm_total / target_count
    teacher_kl = kl_total / max(1, token_count)
    if not math.isfinite(validation_loss) or not math.isfinite(teacher_kl):
        raise ValueError("evaluation produced non-finite validation loss or teacher KL")
    perplexity_overflow = validation_loss > math.log(sys.float_info.max)
    perplexity = None if perplexity_overflow else math.exp(validation_loss)
    return {
        "validation_loss": validation_loss,
        "perplexity": perplexity,
        "perplexity_overflow": perplexity_overflow,
        "teacher_kl": teacher_kl,
        "parameter_count": sum(parameter.numel() for parameter in student.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad),
        "inference_tokens_per_second": token_count / max(inference_seconds, 1e-9),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }


def _save_checkpoint(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def run_pipeline(config: PipelineConfig) -> None:
    """Run iterative importance evaluation, pruning, and repair distillation."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

    validate_pipeline_config(config)
    validate_single_process_environment()
    teacher_config = AutoConfig.from_pretrained(config.teacher_model_path)
    validate_pipeline_config(config, teacher_config.num_hidden_layers)
    output_dir = Path(config.output_dir)
    _prepare_output_dir(output_dir)
    _write_json(output_dir / "run_config.json", asdict(config))
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(config.dtype)
    load_kwargs = {"torch_dtype": dtype}
    if config.attention_implementation:
        load_kwargs["attn_implementation"] = config.attention_implementation
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(config.teacher_model_path, **load_kwargs).to(device)
    student = AutoModelForCausalLM.from_pretrained(config.teacher_model_path, **load_kwargs).to(device)
    _freeze_teacher(teacher)
    importance_texts, train_texts, eval_texts = _load_text_splits(config)
    importance_batches = TextBatcher(importance_texts, tokenizer, config.eval_batch_size, config.max_sequence_length)
    train_batches = TextBatcher(train_texts, tokenizer, config.train_batch_size, config.max_sequence_length, shuffle=True, seed=config.seed)
    eval_batches = TextBatcher(eval_texts, tokenizer, config.eval_batch_size, config.max_sequence_length)
    teacher_metrics = _evaluate(teacher, teacher, eval_batches, config, device)
    _write_json(output_dir / "teacher_metrics.json", teacher_metrics)

    original_layer_ids = list(range(teacher_config.num_hidden_layers))
    all_metrics = []
    iteration = 0
    while len(decoder_layers(student)) > config.target_num_layers:
        iteration += 1
        old_depth = len(decoder_layers(student))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        scores = evaluate_block_influence(student, importance_batches, device)
        ranking = rank_layers(scores, old_depth, config.protect_first_layers, config.protect_last_layers)
        remove_count = min(config.layers_per_iteration, old_depth - config.target_num_layers)
        if len(ranking) < remove_count:
            raise ValueError("protected ranges leave too few eligible layers to reach target_num_layers")
        removed_current = sorted(ranking[:remove_count])
        repair_layers = select_repair_layers(old_depth, removed_current, config.repair_radius)
        original_before = list(original_layer_ids)
        original_layer_ids, removed_original = prune_layers(student, removed_current, original_layer_ids)
        set_trainable_parameters(student, repair_layers, config.train_final_norm, config.train_lm_head)

        iteration_dir = output_dir / "iterations" / f"iteration_{iteration:03d}"
        _save_checkpoint(student, tokenizer, iteration_dir / "pruning_only")
        pruning_only_evaluation = _evaluate(teacher, student, eval_batches, config, device)
        training = _train_iteration(teacher, student, train_batches, original_layer_ids, repair_layers, config, device)
        distilled_evaluation = _evaluate(teacher, student, eval_batches, config, device)
        _save_checkpoint(student, tokenizer, iteration_dir / "distilled")
        metadata = {
            "iteration": iteration,
            "seed": config.seed,
            "depth_before": old_depth,
            "depth_after": len(original_layer_ids),
            "block_influence_scores": {str(index): score for index, score in enumerate(scores)},
            "ranking": ranking,
            "removed_current_layer_indices": removed_current,
            "removed_original_layer_indices": removed_original,
            "original_layer_ids_before": original_before,
            "original_layer_ids": original_layer_ids,
            "repair_layers": repair_layers,
            "training_configuration": asdict(config),
            "training_metrics": training,
            "evaluation_metrics": {"pruning_only": pruning_only_evaluation, "distilled": distilled_evaluation},
        }
        _write_json(iteration_dir / "metadata.json", metadata)
        all_metrics.append(metadata)

    final_dir = output_dir / "final"
    _save_checkpoint(student, tokenizer, final_dir)
    _write_json(output_dir / "training_metrics.json", all_metrics)

    del teacher, student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(final_dir, torch_dtype=dtype, attn_implementation=config.attention_implementation, local_files_only=True).to(device)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(final_dir, local_files_only=True)
    smoke_batch = reloaded_tokenizer(eval_texts[: config.eval_batch_size], padding=True, truncation=True, max_length=config.max_sequence_length, return_tensors="pt")
    with torch.no_grad():
        reload_outputs = reloaded(**_move_batch(smoke_batch, device), use_cache=False)
    if not torch.isfinite(reload_outputs.logits).all():
        raise ValueError("reloaded final checkpoint produced non-finite logits")
    _write_json(output_dir / "reload_validation.json", {"checkpoint": str(final_dir), "model_reload": True, "tokenizer_reload": True, "forward_pass": True})
