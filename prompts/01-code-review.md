# Code review prompt

Review the current branch as a skeptical ML systems maintainer.

Focus on correctness problems rather than formatting. Check layer-index mapping, hidden-state indexing, pruning metadata, padding masks, distributed assumptions, checkpoint reloadability, and whether the tests could pass while the real pipeline is broken.

Run the existing CPU tests, add focused regression tests for confirmed gaps, fix the problems you find, and report each issue with its evidence. Do not modify files under `prompts/` and do not touch `iterative_pruning_distillation.py`.
