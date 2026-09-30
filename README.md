
# Iterative Layer-wise Distillation


<h4 align="center">
   <a> A practical implementation of iterative distillation for compressing Large Language Models (LLMs) </a>
</h4>   

<h4 align="center">
  <a href="">Paper</a> |
  <a href="https://huggingface.co/kaengreg/Qwen2.5-2B-layerwise-distilled">Distilled Models on HuggingFace</a>
</h4>

---

## Overview

This repository contains an implementation of **Iterative Layer-wise Distillation**, a structured approach for distilling LLMs by ranking and removing transformer layers based on their contribution to downstream performance. The approach is inspired by [ShortGPT (2024)](https://arxiv.org/pdf/2403.03853).

The method iteratively prunes layers and fine-tunes the resulting student model using a diverse set of benchmarks covering reasoning, summarization, translation, and generation tasks.
<h1 align="center">
<img style="vertical-align:middle" width="850" height="450" src="https://github.com/kaengreg/layer-wise_distillation/blob/47cce696b4e1b1bc1bdbaff992bec7ab4d8861ba/images/results_eng.png" />
</h1>

---

## Layer importance evaluation

Layer importance is calculated by evalutaing model without target layer on seven datasets from the [LLMTF](https://github.com/RefalMachine/llmtf_open) benchmark:

### MMLU Tasks
- [nlpcoreteam/enMMLU](https://huggingface.co/datasets/NLPCoreTeam/mmlu_en)
- [nlpcoreteam/ruMMLU](https://huggingface.co/datasets/NLPCoreTeam/mmlu_ru)

### Abstractive Summarization
- [dichspace/daru_treeway_eval](https://huggingface.co/datasets/dichspace/daru_treeway_eval)

### Text Copying
- [RefalMachine/cp_doc_ru](https://huggingface.co/datasets/RefalMachine/darumeru/viewer/cp_doc_ru)
- [RefalMachine/cp_para_ru](https://huggingface.co/datasets/RefalMachine/darumeru/viewer/cp_para_ru)

### Machine Translation
- [flores_en_ru](https://huggingface.co/datasets/RefalMachine/darumeru/viewer/flores?views%5B%5D=flores_test)
- [flores_ru_en](https://huggingface.co/datasets/RefalMachine/darumeru/viewer/flores?views%5B%5D=flores_test)

---

## Installation

Clone this repository:

```bash
git clone https://github.com/kaengreg/layer-wise_distillation.git
cd layer-wise_distillation
```

### Option 1: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate layerwise-distillation
```
### Option 2: Using pip

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure to install `torch` and CUDA-specific dependencies manually as needed for your setup.

---

## Usage

### Single-GPU Run

```bash
python3 Qwen2Distillation.py \
    --student_model_path $STUDENT \
   --distil_layers $TRAIN_LAYERS \
   --removed_layers_iterations $PRUNE_LAYERS_1 \
   --removed_layers_iterations $PRUNE_LAYERS_2 \
   --removed_layers_iterations $PRUNE_LAYERS_3 \
   --removed_layers_iterations $PRUNE_LAYERS_4 \
   --learning_rate $LR \
   --num_train_epochs $EPOCHS \
   --per_device_train_batch_size $BS \
   --per_device_eval_batch_size $BS \
   --gradient_accumulation_steps $GRADACM \
   --maxlen $MAXLEN \
   --ds_frac $DSFRAC \
   --use_local_data true \
   --norm_factor $NORMFACT \
   --output_dir $OUTPUT_DIR \
   --logging_dir $LOGGING_DIR
```

### Multi-GPU / Multi-Node via SLURM

Use the following scripts for multi-node, multi-GPU training on SLURM:

- [run_distillation.sh](multinode-multigpu-scripts/run_distillation.sh)
- [run_ft.sh](multinode-multigpu-scripts/run_ft.sh)

> Replace `python3` with `torchrun` for distributed training.

---

## Arguments


| Argument                          | Description |
|----------------------------------|-------------|
| `--teacher_model_path`           | Path to the full teacher model (default: `"Qwen/Qwen2.5-3B"`). |
| `--student_model_path`           | Path to the student model (default: `"kngrg/Qwen2.5-3B-trimmed2"`). |
| `--distil_layers`                | List of layer indices to retain in the student model (e.g., `--distil_layers 0 1 2 3`). |
| `--removed_layers_iterations`    | List(s) of layer indices to be removed at each distillation iteration (e.g., `--removed_layers_iterations 4 5` `--removed_layers_iterations 6 7`). |
| `--train_dataset`                | Name or path of the Hugging Face dataset to use (default: `"kngrg/ru-miracl-cleaned"`). |
| `--learning_rate`                | Learning rate used for optimization (default: `1e-4`). |
| `--num_train_epochs`             | Number of epochs for training per iteration (default: `1`). |
| `--per_device_train_batch_size`  | Batch size per GPU for training (default: `16`). |
| `--per_device_eval_batch_size`   | Batch size per GPU for evaluation (default: `16`). |
| `--gradient_accumulation_steps`  | Number of forward-backward passes before one optimizer step (default: `16`). |
| `--eval_steps`                   | Evaluate the model every N steps (default: `50`). |
| `--save_steps`                   | Save a checkpoint every N steps (default: `512`). |
| `--logging_steps`                | Log training metrics every N steps (default: `1`). |
| `--warmup_steps`                 | Linear warm-up over this many steps (default: `8`). |
| `--max_grad_norm`                | Gradient clipping threshold (default: `0.3`). |
| `--weight_decay`                 | Weight decay coefficient for regularization (default: `0.05`). |
| `--bf16`                         | Enable bfloat16 mixed precision training (default: `True`). |
| `--fp16`                         | Enable float16 mixed precision training (default: `False`). |
| `--maxlen`                       | Maximum token sequence length (default: `512`). |
| `--ds_frac`                      | Number of training samples to use per epoch (default: `3`). |
| `--use_local_data`              | Whether to load dataset from local disk (`true` or `false`, default: `false`). |
| `--norm_factor`                  | Normalization factor applied to the components of the loss function (default: `0.1`). |
| `--output_dir`                   | Directory to save the distilled model checkpoints (default: `./qwen2.5-3b-trimmed2-logits+hs`). |
| `--logging_dir`                  | Directory to store training logs (default: `./logs-logits+hs`). |


---

## Distilled Models

- **[Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) → [Qwen2.5-2B-layerwise-distilled](https://huggingface.co/kaengreg/Qwen2.5-2B-layerwise-distilled)**  
  A reduced-size version of the Qwen2.5-3B model, preserving most of its performance with fewer parameters.

## Technical Report

[TODO] Add a detailed report covering the methodology, pruning strategy, evaluation metrics, and final benchmark results.

## Evaluation

Model quality is evaluated with [LLMTF Open](https://github.com/RefalMachine/llmtf_open). The project uses English and Russian MMLU, Daru Treeway summarization, Russian document and paragraph copying, and FLORES translation in both English-Russian directions.

For reproducible teacher, pruning-only, and distilled-student comparisons, including the required LLMTF revision and saved artifacts, see [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

## Reproducible iterative pipeline

The maintained pipeline recomputes padding-aware Block Influence before every pruning step, removes current student layers while retaining their original teacher indices, and repairs neighboring layers against a frozen original teacher. Legacy scripts above are preserved for reproduction of earlier experiments.

The pipeline intentionally supports one process on one CPU/GPU. Do not launch it through multi-process `torchrun` or multi-task `srun`. The output directory must be new or empty so that checkpoints and metadata from different runs cannot be mixed.

Install the runtime and test dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run CPU tests without Hub access (this is also the GitHub Actions command):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

Basic CLI usage:

```bash
python -m layerwise_distillation.run \
    --teacher_model_path Qwen/Qwen2.5-3B \
    --dataset kngrg/ru-miracl-cleaned \
    --text_column text \
    --target_num_layers 34 \
    --layers_per_iteration 1 \
    --output_dir outputs/qwen2.5-3b-two-layer-pruning
```

### Pipeline arguments

| Argument | Meaning and default |
| --- | --- |
| `--teacher_model_path` | Original frozen teacher and initial student (`Qwen/Qwen2.5-3B`). |
| `--dataset`, `--dataset_split`, `--text_column` | Hugging Face dataset, split, and text field (`kngrg/ru-miracl-cleaned`, `train`, `text`). |
| `--target_num_layers` | Required final decoder depth; must be smaller than the teacher depth. |
| `--layers_per_iteration` | Maximum number removed per iteration (`1`). |
| `--output_dir` | Experiment artifact directory. |
| `--max_importance_samples` | Texts used for each fresh Block Influence pass (`512`). |
| `--max_train_samples`, `--max_eval_samples` | Training and held-out diagnostic limits (`20000`, `1000`). |
| `--max_sequence_length` | Tokenized sequence limit (`512`). |
| `--protect_first_layers`, `--protect_last_layers` | Protected layers at both ends (`1`, `1`). |
| `--repair_radius` | Surviving layers selected on each side of a removed segment (`1`). |
| `--temperature` | KL temperature (`2.0`). |
| `--kl_weight`, `--hidden_weight`, `--lm_weight` | KL, mapped hidden-state MSE, and causal-LM weights (`1.0`, `1.0`, `0.0`). |
| `--max_train_steps` | Optimizer steps per pruning iteration (`200`; `0` skips repair). |
| `--train_batch_size`, `--eval_batch_size` | Per-device batches (`1`, `1`). |
| `--gradient_accumulation_steps` | Micro-batches per optimizer step (`1`). |
| `--learning_rate`, `--weight_decay` | AdamW settings (`1e-5`, `0.0`). |
| `--train_final_norm` / `--no-train_final_norm` | Enable or disable final-norm repair (enabled). |
| `--train_lm_head` / `--no-train_lm_head` | Enable or disable LM-head repair (enabled). |
| `--dtype` | `float32`, `float16`, or `bfloat16` (`bfloat16`). |
| `--attention_implementation` | `eager`, `sdpa`, or `flash_attention_2` (`sdpa`). |
| `--seed` | Training/data seed (`1337`). |
| `--report_to` | `json` or `none`; JSON artifacts are always retained (`json`). |

### One-H100 experiment

Run this exact training command on the remote H100. It performs two one-layer iterations, not one two-layer iteration:

```bash
CUDA_VISIBLE_DEVICES=0 python -m layerwise_distillation.run \
    --teacher_model_path Qwen/Qwen2.5-3B \
    --dataset kngrg/ru-miracl-cleaned \
    --dataset_split train \
    --text_column text \
    --target_num_layers 34 \
    --layers_per_iteration 1 \
    --max_importance_samples 512 \
    --max_sequence_length 512 \
    --max_train_samples 20000 \
    --max_eval_samples 1000 \
    --max_train_steps 200 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --protect_first_layers 1 \
    --protect_last_layers 1 \
    --repair_radius 1 \
    --temperature 2.0 \
    --kl_weight 1.0 \
    --hidden_weight 1.0 \
    --lm_weight 0.0 \
    --dtype bfloat16 \
    --attention_implementation sdpa \
    --seed 1337 \
    --report_to json \
    --output_dir outputs/qwen2.5-3b-two-layer-pruning
```

The recorded diagnostics are validation loss, perplexity, teacher KL, parameter count, peak allocated GPU memory, and inference token throughput. They are not substitutes for downstream evaluation.

### LLMTF Open setup and evaluation

The evaluation wrapper is pinned to LLMTF commit `d36543888b6cc3865cf3a584b5c1bda0b0455567` and rejects any other checkout:

```bash
git clone https://github.com/RefalMachine/llmtf_open.git ../llmtf_open
git -C ../llmtf_open checkout d36543888b6cc3865cf3a584b5c1bda0b0455567
python -m venv ../llmtf-venv
source ../llmtf-venv/bin/activate
python -m pip install -r ../llmtf_open/requirements.txt
```

The smoke mode runs one probability task and one generation task for each of the teacher, final pruning-only checkpoint, and final distilled checkpoint. The full mode runs the required seven tasks for the same three paths. Both commands use foundational prompting, five shots, an 8192-token model context (leaving a 4096-token prompt budget for the document-copy task), deterministic generation, vLLM, one GPU, and isolated output directories:

```bash
CUDA_VISIBLE_DEVICES=0 python -m layerwise_distillation.run_llmtf \
    --llmtf_dir ../llmtf_open \
    --experiment_dir outputs/qwen2.5-3b-two-layer-pruning \
    --mode smoke

CUDA_VISIBLE_DEVICES=0 python -m layerwise_distillation.run_llmtf \
    --llmtf_dir ../llmtf_open \
    --experiment_dir outputs/qwen2.5-3b-two-layer-pruning \
    --mode full
```

Use `--dry_run` to print the three exact underlying LLMTF commands and write `protocol.json` without starting evaluation. The wrapper resolves the final pruning-only checkpoint from `training_metrics.json` instead of assuming a fixed iteration count, verifies that both final checkpoint depths match the metadata, verifies the seven identifiers in the pinned task registry, keeps every original `*_params.jsonl`, result JSONL, `*_total.jsonl`, summary, and log file, then writes `comparison.json` with native per-task metrics. It stops rather than comparing different task sets.

### Artifacts

```text
outputs/qwen2.5-3b-two-layer-pruning/
├── run_config.json
├── teacher_metrics.json
├── training_metrics.json
├── reload_validation.json
├── iterations/
│   ├── iteration_001/{pruning_only,distilled,metadata.json}
│   └── iteration_002/{pruning_only,distilled,metadata.json}
├── final/                         # reloadable final model and tokenizer
└── llmtf/{smoke,full}/
    ├── protocol.json
    ├── teacher/
    ├── pruning_only/
    ├── distilled/
    └── comparison.json
```

Each iteration metadata file contains current and original removed indices, the complete original-layer mapping, fresh Block Influence scores and ranking, repair layers, seed, full training configuration, and pruning-only/distilled diagnostics.
