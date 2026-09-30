# Code Review Report

## Scope

This review treated the iterative pruning pipeline as an ML systems component rather than relying on isolated unit-test success. The reviewed execution path covered importance measurement, iterative layer removal, teacher/student mapping, repair losses, checkpoint creation and reload, diagnostic metrics, and the LLMTF handoff.

## Confirmed correctness issues and resolutions

1. **Causal LM masking depended on padding direction.** The target token alone was masked, so left padding admitted a transition from padding into the first real token. The loss and diagnostic denominator now require valid tokens on both sides of each causal transition.
2. **Hidden-state distillation had an unsafe fallback.** Hugging Face hidden-state tuples do not uniformly represent raw decoder-block outputs at every index. Hidden repair now requires exact block outputs collected from decoder hooks and validates all student-to-teacher indices.
3. **Distributed launches could race on one GPU and artifact directory.** The entry point now rejects multi-process `torchrun`, nonzero local ranks, and multi-task SLURM launches.
4. **Stale artifacts could be mixed with a new run.** A run now requires a new or empty output directory.
5. **Non-finite training could be persisted as success.** Block Influence scores, losses, gradients, evaluation metrics, and reloaded logits are checked for finite values.
6. **Some accepted configurations had no usable gradient path.** Invalid hidden-only repair configurations with no repair layers are rejected before model loading.
7. **Checkpoint validation omitted the tokenizer.** Final validation reloads both model and tokenizer locally and performs a finite forward pass.
8. **LLMTF assumed that iteration 2 was always final.** The wrapper resolves the last completed iteration from metadata, uses absolute paths, and checks that pruning-only and distilled checkpoint depths agree with metadata.
9. **Diagnostic perplexity could silently saturate.** Overflow is now represented explicitly rather than clipping the validation loss before exponentiation.

## Scientific and systems boundaries

- The teacher is frozen once and is never placed in training mode.
- Current student indices and original teacher indices are recorded separately at every iteration.
- Block Influence is recomputed on the current student before each pruning operation.
- The implementation deliberately supports one process and one GPU; multi-node training is out of scope.
- Validation loss, perplexity, and teacher KL remain diagnostics and are not treated as downstream-quality evidence.

## Verification performed

During the code-review pass, the offline CPU suite reached 26 passing tests after the correctness fixes. The subsequent test-review commit expands this to 29 tests and records the final command and result separately.

No H100, CUDA, live model, live dataset, or live LLMTF evaluation was run. Those checks remain pending.
