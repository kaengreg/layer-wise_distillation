# Task: Reproducible Iterative Layer Pruning and Distillation

## Context

The repository currently contains standalone scripts for Qwen2.5 layer importance evaluation, layer removal, and knowledge distillation.

The existing implementation is difficult to test because model loading, pruning, loss calculation, training, and filesystem operations are tightly coupled. Some scripts also contain hard-coded paths and layer indices.

The untracked file `iterative_pruning_distillation.py` is not an input to this task. Do not read from it, copy it, import it, modify it, delete it, or add it to git.

## Goal

Implement a reproducible iterative pipeline that:

1. evaluates the importance of the current student layers;
2. removes the least important eligible layers;
3. selects neighboring layers for repair;
4. distils the pruned student from the unchanged original teacher;
5. saves a reloadable checkpoint and iteration metadata;
6. repeats until the requested target depth is reached.

The implementation must be testable on CPU without downloading pretrained models. A separate command must support a real experiment on one NVIDIA H100.

## Functional requirements

### Layer importance

- Implement Block Influence using cosine similarity between each block input and output.
- Ignore padding tokens.
- Average scores over sequences rather than over padded tensor positions.
- Recompute scores before every pruning iteration.
- Resolve equal scores deterministically by current layer index.
- Allow the first and last N layers to be protected.

### Layer pruning

- Remove layers by their current student indices.
- Preserve `original_layer_ids`, mapping every current student layer to the corresponding layer in the original teacher.
- Update `config.num_hidden_layers`.
- Update attention `layer_idx` values where present.
- Update architecture metadata such as `layer_types` or `max_window_layers` when present.
- Reject removal of all layers and out-of-range layer indices.

### Repair and distillation

- Freeze the teacher permanently.
- Freeze student parameters except selected repair layers, final norm, and LM head when enabled.
- Select repair layers around each contiguous removed segment.
- Implement temperature-scaled token-level KL divergence, hidden-state MSE against mapped original teacher layers, and optional causal language-model loss.
- Mask padding tokens in every applicable loss.
- Expose the weights of all loss components through CLI arguments.

### Checkpoints and metadata

For every iteration save:

- the student model and tokenizer;
- current and original removed layer indices;
- remaining `original_layer_ids`;
- Block Influence scores;
- selected repair layers;
- training configuration;
- evaluation metrics;
- random seed.

Also save a final checkpoint that can be loaded through `AutoModelForCausalLM.from_pretrained`.

## Interface

Provide an entry point runnable as:

```bash
python -m layerwise_distillation.run \
    --teacher_model_path Qwen/Qwen2.5-3B \
    --dataset kngrg/ru-miracl-cleaned \
    --text_column text \
    --target_num_layers 34 \
    --layers_per_iteration 1 \
    --output_dir outputs/qwen2.5-3b-two-layer-pruning
```

The CLI must also expose sample limits, sequence length, protected layers, repair radius, temperature, loss weights, training steps, batch sizes, dtype, attention implementation, seed, and reporting backend.

Use `argparse` unless there is a strong repository-specific reason to choose another dependency.

## H100 experiment

Provide a documented command or shell script with these defaults:

- teacher: `Qwen/Qwen2.5-3B`;
- remove two layers in two iterations;
- dataset: `kngrg/ru-miracl-cleaned`;
- Block Influence samples: 512;
- maximum sequence length: 512;
- maximum training samples: 20,000;
- maximum evaluation samples: 1,000;
- maximum training steps: 200 per iteration;
- dtype: BF16;
- one GPU;
- fixed seed: 1337.

Compare:

1. the original teacher;
2. the student immediately after pruning;
3. the student after repair distillation.

Report validation loss, perplexity, KL divergence to the teacher, parameter count, peak GPU memory, and inference throughput.

The experiment is successful technically when the pipeline completes and the checkpoint reloads. Do not require that a short distillation run always improves every quality metric.

## Tests

Add CPU tests covering:

- deterministic layer ranking and protected ranges;
- pruning one and several layers;
- original layer mapping across multiple iterations;
- configuration and attention-index updates;
- repair-layer selection around isolated and contiguous removals;
- padding masks and temperature scaling in the losses;
- invalid target depth and invalid layer indices;
- save, reload, and forward pass of a tiny randomly initialized Qwen2 model;
- one finite backward and optimizer step.

Tests must not access Hugging Face Hub or require CUDA.

## Dependency and documentation cleanup

- Replace the current non-portable dependency dump with minimal runtime and development dependency files.
- Add a CPU test command suitable for GitHub Actions.
- Update the README with installation, CLI usage, tests, H100 execution, and output artifact descriptions.
- Preserve the old research scripts unless a small compatibility change is necessary and documented.

## Out of scope

- Full LLMTF evaluation.
- Multi-node training.
- Production deployment.
- Support for every Transformers architecture.
- Full convergence of the 3B student.

## Acceptance criteria

The task is complete when:

- all CPU tests pass;
- the test suite performs no network access;
- the tiny-model checkpoint reloads and runs a forward pass;
- the CLI validates its arguments before loading large models;
- each iteration produces model and metadata artifacts;
- the README contains an exact H100 command;
- skipped GPU verification is explicitly reported;
- `iterative_pruning_distillation.py` remains untouched and untracked.
