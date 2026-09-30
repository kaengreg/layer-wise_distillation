# Repository Instructions

## Project purpose

This repository contains research code for iterative layer-wise pruning and knowledge distillation of decoder-only language models, primarily Qwen2.5.

Prefer clear, reproducible research code over production-oriented abstractions.

## Working rules

- Inspect the relevant code and current git status before making changes.
- Preserve unrelated user changes and untracked files.
- Do not modify legacy scripts unless the active task explicitly requires it.
- Keep GPU-specific execution separate from CPU-testable algorithmic logic.
- Do not download model weights or datasets during unit tests.
- Do not claim that a GPU experiment succeeded unless its logs and output artifacts were actually inspected.
- Use deterministic seeds where practical.
- Keep generated checkpoints, datasets, logs, and experiment outputs out of git.

## ML correctness requirements

- Keep the original teacher model frozen throughout distillation.
- Recompute layer importance after every pruning iteration.
- Preserve the mapping between current student layers and original teacher layer indices.
- Update model configuration and attention layer indices after pruning.
- Exclude padding tokens from KL, hidden-state, and language-model losses.
- Save enough metadata to reproduce every pruning decision.
- Validate that saved student checkpoints can be loaded again.

## Code style

- Keep function arguments on one line when the result remains readable.
- Prefer small task-specific functions over unnecessary class hierarchies.
- Add explicit exceptions only for important invalid states and user-facing configuration errors.
- Use English for code, identifiers, docstrings, and technical messages.
- Avoid speculative generalization beyond decoder models exposing `model.layers`.

## Verification

- Add CPU unit tests for algorithmic code.
- Use tiny randomly initialized models in tests.
- Run the relevant tests after each meaningful change.
- Report exactly which checks were run and which were skipped.
- A skipped H100 experiment must be described as pending, not passed.
- Treat validation loss, perplexity, and teacher KL as diagnostic metrics, not as substitutes for downstream evaluation.
- Use LLMTF Open for the final downstream comparison of the teacher, pruning-only student, and distilled student.
- Keep the LLMTF revision, task list, conversation config, few-shot count, generation settings, and sample limits identical across compared checkpoints.
- Record the LLMTF git commit and complete evaluation command with every benchmark result.

## Documentation

- Keep public CLI arguments documented in the README.
- Record experiment parameters and results in machine-readable JSON.
- Follow `docs/evaluation-protocol.md` when preparing or analyzing the H100 experiment.
- Do not edit or fabricate files under `prompts/`; they contain prompts written by me.
