#!/bin/bash

#SBATCH --nodes=1                   
#SBATCH --gres=gpu:8       
#SBATCH --cpus-per-task=128
#SBATCH --time=2:00:00

export NNODES=1
export GPUS_PER_NODE=8
export BATCH_SIZE=8
export CONV_PATH="conversation_configs/non_instruct_simple.json"
export FS_COUNT=5

export MODEL_TO_EVAL=/scratch/s02210660/diploma-llm/qwen2.5-ft_512/qwen2.5-3b-trimmed2-logits+hs/student
export MODEL_TO_EVAL_OUTPUT_DIR=/scratch/s02210660/diploma-llm/qwen2.5-ft_512/qwen2.5-3b-trimmed2-logits+hs/student/llmtf-eval_res_k5

echo $MODEL_TO_EVAL
echo $MODEL_TO_EVAL_OUTPUT_DIR

srun --nodes=1 --gpus 8 --cpus-per-task 128 -o "/scratch/s02210660/diploma-llm/llmtf_evals_logs/slurm-%A.out" \
     --container-image /scratch/practical_llm/distill/ngc_cuda_pytorch_vllm_20_10_24_v8.sqsh \
     --container-workdir /scratch/s02210660/diploma-llm \
     --container-mounts /scratch/s02210660/diploma-llm:/scratch/s02210660/diploma-llm \
     bash -c "cd /scratch/s02210660/diploma-llm/llmtf_open && ./run_evaluate_multinode_multigpu_universal.sh" & 
