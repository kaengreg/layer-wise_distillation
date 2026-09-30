import json

import pytest

from layerwise_distillation.aggregate_llmtf import aggregate_results


def write_total(directory, task, metric):
    directory.mkdir()
    (directory / f"{task}_total.jsonl").write_text(json.dumps({"accuracy": metric}) + "\n", encoding="utf-8")


def test_aggregate_preserves_native_metrics(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"llmtf_commit": "abc123"}), encoding="utf-8")
    directories = {role: tmp_path / role for role in ("teacher", "pruning_only", "distilled")}
    for index, (role, directory) in enumerate(directories.items()):
        write_total(directory, "en_mmlu", index / 10)
    result = aggregate_results(directories, protocol)
    assert result["llmtf_commit"] == "abc123"
    assert result["rows"] == [{"task": "en_mmlu", "teacher": {"accuracy": 0.0}, "pruning_only": {"accuracy": 0.1}, "distilled": {"accuracy": 0.2}}]


def test_aggregate_reads_pretty_multiline_llmtf_totals(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"llmtf_commit": "abc123"}), encoding="utf-8")
    directories = {role: tmp_path / role for role in ("teacher", "pruning_only", "distilled")}
    for directory in directories.values():
        directory.mkdir()
        payload = {"task_name": "nlpcoreteam/enMMLU", "results": {"acc": 0.5}}
        (directory / "nlpcoreteam_enMMLU_total.jsonl").write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    result = aggregate_results(directories, protocol)
    assert result["rows"][0]["task"] == "nlpcoreteam/enMMLU"


def test_aggregate_rejects_different_task_sets(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"llmtf_commit": "abc123"}), encoding="utf-8")
    directories = {role: tmp_path / role for role in ("teacher", "pruning_only", "distilled")}
    write_total(directories["teacher"], "task_a", 1)
    write_total(directories["pruning_only"], "task_b", 1)
    write_total(directories["distilled"], "task_a", 1)
    with pytest.raises(ValueError, match="task sets differ"):
        aggregate_results(directories, protocol)
