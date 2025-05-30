
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

