# LLMTF Evaluation Protocol

## Purpose

The final quality assessment uses [LLMTF Open](https://github.com/RefalMachine/llmtf_open). Validation loss, perplexity, KL divergence, parameter count, memory use, and throughput are useful diagnostics, but they do not replace downstream evaluation.

The protocol compares three checkpoints under identical conditions:

1. the original `Qwen/Qwen2.5-3B` teacher;
2. the final-depth student immediately after pruning and before repair distillation;
3. the final-depth student after repair distillation.

The pipeline must therefore preserve the pruning-only checkpoint in addition to the post-distillation checkpoint.

## Benchmark suite

Use the same seven evaluations described by the original project:

| Capability | Dataset or direction |
| --- | --- |
| English knowledge and reasoning | `nlpcoreteam/enMMLU` |
| Russian knowledge and reasoning | `nlpcoreteam/ruMMLU` |
| Abstractive summarization | `dichspace/daru_treeway_eval` |
| Russian document copying | `RefalMachine/darumeru`, `cp_doc_ru` subset |
| Russian paragraph copying | `RefalMachine/darumeru`, `cp_para_ru` subset |
| Machine translation | FLORES English to Russian |
| Machine translation | FLORES Russian to English |

LLMTF task identifiers can change between revisions. Before preparing the run command, inspect the task registry of the checked-out LLMTF revision and map these seven evaluations to its actual identifiers. If an evaluation is unavailable, stop and report it rather than substituting a different task silently.

## Reproducibility rules

- Pin LLMTF Open to a specific git commit and record `git rev-parse HEAD`.
- Keep LLMTF in a separate checkout or environment; do not vendor it into this repository.
- Treat Qwen2.5-3B as a foundational/base model. Use the foundational conversation configuration and corresponding LLMTF flag supported by the pinned revision.
- Use five-shot evaluation to match the historical project scripts.
- Keep the task list, conversation config, few-shot count, prompt length, sample limits, batch size, backend, generation config, tokenizer, and random seed identical for all three checkpoints.
- Give every checkpoint a separate output directory. Never allow one model's cached task results to be reused for another model.
- Use `--force_recalc` or the equivalent when rerunning an existing output directory.
- Preserve LLMTF's original per-example results, per-task parameter files, aggregate files, summary table, and evaluation log.
- Report failures and missing tasks explicitly. Do not compute a mean over different task sets.

## Two evaluation stages

### Smoke evaluation

Run a small deterministic subset for all three checkpoints before the full benchmark. Its purpose is to verify:

- checkpoint loading
- foundational prompt formatting
- successful execution of at least one probability-based and one generation-based task
- creation and parsing of LLMTF aggregate result files
- separation of output directories

Smoke results are integration evidence, not final model-quality results.

### Final evaluation

Run the complete seven-task suite for all three checkpoints using the frozen configuration. Produce a comparison table containing every native LLMTF metric for each task. Do not collapse heterogeneous metrics such as accuracy, ROUGE, BLEU, and copy-task scores into an undocumented average.

If an aggregate score is added, document its normalization and directionality and retain the per-task metrics as the primary result.

## Required artifacts

Store experiment artifacts outside git. The implementation must document a layout equivalent to:

```text
outputs/<experiment>/
├── run_config.json
├── training_metrics.json
├── checkpoints/
│   ├── pruning_only/
│   └── distilled/
└── llmtf/
    ├── protocol.json
    ├── teacher/
    ├── pruning_only/
    ├── distilled/
    └── comparison.json
```

`protocol.json` must contain:

- LLMTF repository URL and git commit
- evaluated model paths and their role
- resolved LLMTF task identifiers
- dataset names and configurations
- conversation config
- few-shot count
- prompt and generation limits
- backend and batch size
- seed
- exact commands

`comparison.json` must preserve each task's native metric names and values for all three models. A Markdown table may be generated from this file for the report.

## Execution responsibility

The coding agent prepares and validates the commands and result parser but does not claim the H100 benchmark has passed without inspecting real LLMTF artifacts. The user runs the GPU experiment. Afterward, the agent validates the files and updates the report from measured results only.

## Pinned executable protocol

The implementation pins LLMTF Open to commit `d36543888b6cc3865cf3a584b5c1bda0b0455567`. At that revision, the required registry identifiers are:

- `nlpcoreteam/enmmlu`
- `nlpcoreteam/rummlu`
- `daru/treewayabstractive`
- `darumeru/cp_doc_ru`
- `darumeru/cp_para_ru`
- `darumeru/flores_en_ru`
- `darumeru/flores_ru_en`

Run `python -m layerwise_distillation.run_llmtf --llmtf_dir ../llmtf_open --experiment_dir outputs/qwen2.5-3b-two-layer-pruning --mode smoke` first, then repeat with `--mode full`. The wrapper validates the commit and task registry, evaluates all three model roles with identical settings, records the exact expanded commands in `protocol.json`, and creates `comparison.json` from LLMTF's native per-task aggregate files. Full setup and commands are in the README.
