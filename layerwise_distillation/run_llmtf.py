"""Run the pinned LLMTF comparison without mixing model artifacts."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from .aggregate_llmtf import aggregate_results


LLMTF_COMMIT = "d36543888b6cc3865cf3a584b5c1bda0b0455567"
FULL_TASKS = [
    "nlpcoreteam/enmmlu",
    "nlpcoreteam/rummlu",
    "daru/treewayabstractive",
    "darumeru/cp_doc_ru",
    "darumeru/cp_para_ru",
    "darumeru/flores_en_ru",
    "darumeru/flores_ru_en",
]
SMOKE_TASKS = ["nlpcoreteam/enmmlu", "daru/treewayabstractive"]


def _git_commit(llmtf_dir: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=llmtf_dir, text=True).strip()


def _validate_registry(llmtf_dir: Path, tasks: list[str]) -> None:
    registry = (llmtf_dir / "llmtf" / "tasks" / "__init__.py").read_text(encoding="utf-8").lower()
    missing = [task for task in tasks if f"'{task.lower()}'" not in registry]
    if missing:
        raise ValueError(f"pinned LLMTF registry is missing required tasks: {missing}")


def _resolve_final_pruning_checkpoint(experiment_dir: Path) -> Path:
    metrics_path = experiment_dir / "training_metrics.json"
    if not metrics_path.is_file():
        raise ValueError(f"missing completed pipeline metadata: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("training_metrics.json contains no completed pruning iterations")
    last_record = max(metrics, key=lambda record: int(record["iteration"]))
    last_iteration = int(last_record["iteration"])
    checkpoint = experiment_dir / "iterations" / f"iteration_{last_iteration:03d}" / "pruning_only"
    checkpoint_config_path = checkpoint / "config.json"
    final_config_path = experiment_dir / "final" / "config.json"
    if not checkpoint_config_path.is_file():
        raise ValueError(f"final pruning-only checkpoint is incomplete: {checkpoint}")
    if not final_config_path.is_file():
        raise ValueError(f"final distilled checkpoint is incomplete: {experiment_dir / 'final'}")
    checkpoint_depth = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))["num_hidden_layers"]
    final_depth = json.loads(final_config_path.read_text(encoding="utf-8"))["num_hidden_layers"]
    recorded_depth = int(last_record["depth_after"])
    if checkpoint_depth != recorded_depth or final_depth != recorded_depth:
        raise ValueError(f"final checkpoint depths disagree with iteration metadata: pruning={checkpoint_depth}, distilled={final_depth}, metadata={recorded_depth}")
    return checkpoint


def _resolve_local_checkpoint(value: str | None, default: Path) -> str:
    checkpoint = Path(value).expanduser().resolve() if value else default.resolve()
    if not (checkpoint / "config.json").is_file():
        raise ValueError(f"checkpoint is missing config.json: {checkpoint}")
    return str(checkpoint)


def _command(llmtf_dir: Path, model: str, output_dir: Path, tasks: list[str], sample_limit: int, batch_size: int) -> list[str]:
    return [
        "python", str(llmtf_dir / "evaluate_model.py"),
        "--model_name_or_path", model,
        "--conv_path", str(llmtf_dir / "conversation_configs" / "default_foundational.json"),
        "--output_dir", str(output_dir),
        "--dataset_names", *tasks,
        "--few_shot_count", "5",
        "--max_sample_per_dataset", str(sample_limit),
        "--batch_size", str(batch_size),
        "--model_context_len", "8192",
        "--temperature", "0.0",
        "--repetition_penalty", "1.0",
        "--vllm", "--tensor_parallel_size", "1", "--is_foundational", "--force_recalc",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run identical pinned LLMTF settings for teacher, pruning-only, and distilled checkpoints")
    parser.add_argument("--llmtf_dir", type=Path, required=True)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--teacher_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--pruning_only_model", default=None)
    parser.add_argument("--distilled_model", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch_size must be positive")
    llmtf_dir = args.llmtf_dir.expanduser().resolve()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    commit = _git_commit(llmtf_dir)
    if commit != LLMTF_COMMIT:
        raise SystemExit(f"LLMTF commit mismatch: expected {LLMTF_COMMIT}, found {commit}")
    tasks = SMOKE_TASKS if args.mode == "smoke" else FULL_TASKS
    sample_limit = 8 if args.mode == "smoke" else 1000
    _validate_registry(llmtf_dir, tasks)
    teacher_path = Path(args.teacher_model).expanduser()
    teacher_model = str(teacher_path.resolve()) if teacher_path.exists() else args.teacher_model
    default_pruning_checkpoint = _resolve_final_pruning_checkpoint(experiment_dir) if args.pruning_only_model is None else experiment_dir
    models = {
        "teacher": teacher_model,
        "pruning_only": _resolve_local_checkpoint(args.pruning_only_model, default_pruning_checkpoint),
        "distilled": _resolve_local_checkpoint(args.distilled_model, experiment_dir / "final"),
    }
    root = experiment_dir / "llmtf" / args.mode
    commands = {role: _command(llmtf_dir, model, root / role, tasks, sample_limit, args.batch_size) for role, model in models.items()}
    protocol = {
        "llmtf_repository": "https://github.com/RefalMachine/llmtf_open.git",
        "llmtf_commit": LLMTF_COMMIT,
        "mode": args.mode,
        "models": models,
        "resolved_task_identifiers": tasks,
        "datasets": {
            "nlpcoreteam/enmmlu": "RefalMachine/darumeru:mmlu_nlpcoreteam (English fields)",
            "nlpcoreteam/rummlu": "RefalMachine/darumeru:mmlu_nlpcoreteam (Russian fields)",
            "daru/treewayabstractive": "dichspace/daru_treeway_eval",
            "darumeru/cp_doc_ru": "RefalMachine/darumeru:cp_doc_ru",
            "darumeru/cp_para_ru": "RefalMachine/darumeru:cp_para_ru",
            "darumeru/flores_en_ru": "RefalMachine/darumeru:flores",
            "darumeru/flores_ru_en": "RefalMachine/darumeru:flores",
        },
        "conversation_config": "conversation_configs/default_foundational.json",
        "is_foundational": True,
        "few_shot_count": 5,
        "model_context_length": 8192,
        "max_sample_per_dataset": sample_limit,
        "batch_size": args.batch_size,
        "backend": "vllm",
        "tensor_parallel_size": 1,
        "generation": {"temperature": 0.0, "repetition_penalty": 1.0, "num_return_sequences": 1},
        "llmtf_internal_seed": 555,
        "commands": {role: shlex.join(command) for role, command in commands.items()},
    }
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        for role, command in commands.items():
            print(f"{role}: {shlex.join(command)}")
        return

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONHASHSEED"] = "555"
    for role, command in commands.items():
        print(f"Running {role}: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=llmtf_dir, env=environment, check=True)
    comparison = aggregate_results({role: root / role for role in models}, protocol_path)
    (root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
