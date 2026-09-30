# Test Review Report

## Coverage audit against `TASK.md`

The pre-audit suite covered the named arithmetic and pruning cases but left several acceptance behaviors weak or observable only through small isolated tests. The review added the smallest behavioral coverage for the highest-risk gaps.

## Added regression coverage

- **Network prohibition:** an autouse fixture fails on DNS resolution and IPv4/IPv6 connection attempts. Hugging Face offline flags are also set before imports.
- **Real iterative execution:** a local randomly initialized Qwen2 model and local tokenizer run through two complete pruning/distillation iterations on CPU.
- **Importance recomputation:** the integration test observes importance evaluation at depths 4 and 3, proving that the second decision uses the already-pruned student.
- **Teacher immutability:** the integration test verifies that all teacher parameters stay frozen and byte-for-byte unchanged through training.
- **Pruning timing:** pruning-only and distilled checkpoints from the same final iteration are verified to differ after the optimizer step.
- **Metadata contract:** every iteration JSON is compared with the aggregate metadata and checked for mapping, scores, repair layers, configuration, seed, and both diagnostic metric groups.
- **Checkpoint reloadability:** every pruning-only, distilled, and final checkpoint reloads both model and tokenizer locally and produces finite logits.
- **Student freezing:** tests verify repair-layer-only training and the optional inclusion of final norm and LM head.
- **Dataset schema boundary:** the real split/filter/limit logic runs against an in-memory `datasets.Dataset`; only the external `load_dataset` boundary is replaced. Missing text columns fail explicitly.
- **Validation ordering:** invalid configuration is rejected before model lookup or output-directory creation.

## Requirements still dependent on external systems

- The current live schema and availability of `kngrg/ru-miracl-cleaned` are not queried by the offline suite.
- Qwen2.5-3B pretrained weights are not downloaded; tiny random Qwen2 models are used.
- CUDA, BF16, Flash Attention, H100 memory, and H100 throughput are pending.
- Live vLLM/LLMTF execution and the seven downstream datasets are pending. Command construction, task-set consistency, and aggregate parsing are CPU-tested with local fixtures.

## Final verification

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

Result:

```text
.............................                                            [100%]
29 passed in 2.83s
```

Additional checks:

```bash
python -m compileall -q layerwise_distillation tests
git diff --check
```

Both completed successfully.
