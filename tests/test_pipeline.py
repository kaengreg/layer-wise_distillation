import json

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

import layerwise_distillation.pipeline as pipeline
from layerwise_distillation.pipeline import PipelineConfig, _load_text_splits, run_pipeline


def create_local_tiny_qwen(path):
    path.mkdir()
    vocabulary = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "one": 3, "two": 4, "three": 5, "four": 6, "five": 7}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.save_pretrained(path)
    config = Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32, tie_word_embeddings=False)
    Qwen2ForCausalLM(config).save_pretrained(path)


def test_two_iteration_pipeline_metadata_and_every_checkpoint_reload(tmp_path, monkeypatch):
    torch.manual_seed(19)
    model_dir = tmp_path / "teacher"
    output_dir = tmp_path / "output"
    create_local_tiny_qwen(model_dir)
    importance = ["one two three", "two three four"]
    training = ["one two three", "two three four", "three four five"]
    evaluation = ["one two three", "two three four"]
    monkeypatch.setattr(pipeline, "_load_text_splits", lambda config: (importance, training, evaluation))
    importance_depths = []
    original_importance = pipeline.evaluate_block_influence

    def track_importance(model, batches, device):
        importance_depths.append(len(model.model.layers))
        return original_importance(model, batches, device)

    teacher_evidence = {}
    original_freeze_teacher = pipeline._freeze_teacher

    def track_teacher(teacher):
        original_freeze_teacher(teacher)
        teacher_evidence["model"] = teacher
        teacher_evidence["state"] = {name: tensor.detach().clone() for name, tensor in teacher.state_dict().items()}

    monkeypatch.setattr(pipeline, "evaluate_block_influence", track_importance)
    monkeypatch.setattr(pipeline, "_freeze_teacher", track_teacher)
    config = PipelineConfig(teacher_model_path=str(model_dir), dataset="offline", text_column="text", target_num_layers=2, layers_per_iteration=1, output_dir=str(output_dir), max_importance_samples=2, max_train_samples=3, max_eval_samples=2, max_sequence_length=8, protect_first_layers=0, protect_last_layers=0, repair_radius=1, max_train_steps=1, train_batch_size=2, eval_batch_size=2, dtype="float32", attention_implementation="eager")
    run_pipeline(config)

    metadata = json.loads((output_dir / "training_metrics.json").read_text(encoding="utf-8"))
    mapping = list(range(4))
    assert len(metadata) == 2
    assert importance_depths == [4, 3]
    teacher = teacher_evidence["model"]
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(torch.equal(tensor, teacher.state_dict()[name]) for name, tensor in teacher_evidence["state"].items())
    for iteration in metadata:
        iteration_path = output_dir / "iterations" / f"iteration_{iteration['iteration']:03d}"
        assert json.loads((iteration_path / "metadata.json").read_text(encoding="utf-8")) == iteration
        removed_current = iteration["removed_current_layer_indices"]
        expected_removed_original = [mapping[index] for index in removed_current]
        mapping = [original for index, original in enumerate(mapping) if index not in set(removed_current)]
        assert iteration["removed_original_layer_indices"] == expected_removed_original
        assert iteration["original_layer_ids"] == mapping
        assert iteration["training_metrics"]["steps"] == 1
        assert set(iteration) >= {"block_influence_scores", "repair_layers", "training_configuration", "evaluation_metrics", "seed"}
        assert set(iteration["evaluation_metrics"]) == {"pruning_only", "distilled"}
        for role in ("pruning_only", "distilled"):
            assert set(iteration["evaluation_metrics"][role]) >= {"validation_loss", "perplexity", "teacher_kl", "parameter_count", "peak_gpu_memory_bytes", "inference_tokens_per_second"}

    checkpoint_dirs = sorted((output_dir / "iterations").glob("iteration_*/*")) + [output_dir / "final"]
    for checkpoint in checkpoint_dirs:
        if not checkpoint.is_dir():
            continue
        reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
        reloaded_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        tokenized = reloaded_tokenizer("one two three", return_tensors="pt")
        output = reloaded(**tokenized)
        assert torch.isfinite(output.logits).all()
        assert len(reloaded.config.layer_types) == reloaded.config.num_hidden_layers
    reload_validation = json.loads((output_dir / "reload_validation.json").read_text(encoding="utf-8"))
    assert reload_validation["model_reload"] is True
    assert reload_validation["tokenizer_reload"] is True
    assert reload_validation["forward_pass"] is True

    pruning_only = AutoModelForCausalLM.from_pretrained(output_dir / "iterations" / "iteration_002" / "pruning_only", local_files_only=True)
    distilled = AutoModelForCausalLM.from_pretrained(output_dir / "iterations" / "iteration_002" / "distilled", local_files_only=True)
    assert any(not torch.equal(pruning_only.state_dict()[name], tensor) for name, tensor in distilled.state_dict().items())


def test_dataset_schema_filtering_and_sample_limits(monkeypatch, tmp_path):
    dataset = Dataset.from_dict({"body": [None, "", "alpha", "beta", "gamma", "delta", "epsilon", "zeta"], "ignored": list(range(8))})
    calls = []

    def local_dataset_loader(name, split):
        calls.append((name, split))
        return dataset

    monkeypatch.setattr("datasets.load_dataset", local_dataset_loader)
    config = PipelineConfig(teacher_model_path="unused", dataset="local-fixture", dataset_split="validation", text_column="body", target_num_layers=2, layers_per_iteration=1, output_dir=str(tmp_path), max_importance_samples=2, max_train_samples=4, max_eval_samples=2, dtype="float32")
    importance, training, evaluation = _load_text_splits(config)
    assert calls == [("local-fixture", "validation")]
    assert len(importance) <= 2
    assert len(training) <= 4
    assert len(evaluation) <= 2
    assert all(text.strip() for text in importance + training + evaluation)

    invalid_config = PipelineConfig(**{**config.__dict__, "text_column": "missing"})
    with pytest.raises(ValueError, match="available columns"):
        _load_text_splits(invalid_config)


def test_invalid_configuration_stops_before_model_lookup(monkeypatch, tmp_path):
    model_lookup_attempted = False

    def forbidden_model_lookup(*args, **kwargs):
        nonlocal model_lookup_attempted
        model_lookup_attempted = True
        raise AssertionError("model lookup must not run")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", forbidden_model_lookup)
    config = PipelineConfig(teacher_model_path="remote/model", dataset="remote/dataset", text_column="text", target_num_layers=0, layers_per_iteration=1, output_dir=str(tmp_path / "output"), dtype="float32")
    with pytest.raises(ValueError, match="target_num_layers"):
        run_pipeline(config)
    assert not model_lookup_attempted
    assert not (tmp_path / "output").exists()
