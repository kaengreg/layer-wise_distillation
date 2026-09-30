from pathlib import Path

import pytest

import json

from layerwise_distillation.run_llmtf import FULL_TASKS, LLMTF_COMMIT, SMOKE_TASKS, _command, _resolve_final_pruning_checkpoint, _validate_registry


def test_llmtf_task_sets_and_command_are_frozen(tmp_path):
    assert len(FULL_TASKS) == 7
    assert set(SMOKE_TASKS).issubset(FULL_TASKS)
    command = _command(Path("/llmtf"), "/model", tmp_path, SMOKE_TASKS, 8, 4)
    assert command[0:2] == ["python", "/llmtf/evaluate_model.py"]
    assert command[command.index("--few_shot_count") + 1] == "5"
    assert command[command.index("--model_context_len") + 1] == "8192"
    assert command[command.index("--max_sample_per_dataset") + 1] == "8"
    assert "--is_foundational" in command
    assert "--vllm" in command
    assert len(LLMTF_COMMIT) == 40


def test_registry_validation_rejects_missing_task(tmp_path):
    registry = tmp_path / "llmtf" / "tasks"
    registry.mkdir(parents=True)
    (registry / "__init__.py").write_text("TASK_REGISTRY = {'known/task': {}}", encoding="utf-8")
    _validate_registry(tmp_path, ["known/task"])
    with pytest.raises(ValueError, match="missing required tasks"):
        _validate_registry(tmp_path, ["missing/task"])


def test_final_pruning_checkpoint_is_resolved_from_metadata_not_fixed_iteration(tmp_path):
    metrics = [{"iteration": 1, "depth_after": 3}, {"iteration": 3, "depth_after": 2}]
    (tmp_path / "training_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    checkpoint = tmp_path / "iterations" / "iteration_003" / "pruning_only"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"num_hidden_layers": 2}), encoding="utf-8")
    final = tmp_path / "final"
    final.mkdir()
    (final / "config.json").write_text(json.dumps({"num_hidden_layers": 2}), encoding="utf-8")
    assert _resolve_final_pruning_checkpoint(tmp_path) == checkpoint
