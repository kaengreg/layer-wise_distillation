"""Aggregate native LLMTF per-task totals across the three comparison roles."""

import argparse
import json
from pathlib import Path
from typing import Any


ROLES = ("teacher", "pruning_only", "distilled")


def _read_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"aggregate file is empty: {path}")
    return records[0] if len(records) == 1 else records


def collect_role_results(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("*_total.jsonl"))
    if not files:
        raise ValueError(f"no LLMTF *_total.jsonl files found in {directory}")
    results = {}
    for path in files:
        payload = _read_jsonl(path)
        task_name = payload.get("task_name") if isinstance(payload, dict) else None
        results[task_name or path.name[: -len("_total.jsonl")]] = payload
    return results


def aggregate_results(role_directories: dict[str, Path], protocol_path: Path) -> dict[str, Any]:
    with protocol_path.open(encoding="utf-8") as stream:
        protocol = json.load(stream)
    results = {role: collect_role_results(role_directories[role]) for role in ROLES}
    task_sets = {role: set(tasks) for role, tasks in results.items()}
    if len({frozenset(tasks) for tasks in task_sets.values()}) != 1:
        details = "; ".join(f"{role}={sorted(tasks)}" for role, tasks in task_sets.items())
        raise ValueError(f"LLMTF task sets differ across roles: {details}")
    rows = []
    for task in sorted(task_sets["teacher"]):
        rows.append({"task": task, **{role: results[role][task] for role in ROLES}})
    return {"llmtf_commit": protocol["llmtf_commit"], "protocol": str(protocol_path), "roles": {role: str(role_directories[role]) for role in ROLES}, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate LLMTF native task metrics without modifying source artifacts")
    parser.add_argument("--teacher_dir", type=Path, required=True)
    parser.add_argument("--pruning_only_dir", type=Path, required=True)
    parser.add_argument("--distilled_dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    directories = {"teacher": args.teacher_dir, "pruning_only": args.pruning_only_dir, "distilled": args.distilled_dir}
    comparison = aggregate_results(directories, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
