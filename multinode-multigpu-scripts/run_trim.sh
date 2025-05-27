#!/bin/bash
#SBATCH --job-name=model_prune
#SBATCH --nodes=1         
#SBATCH --gres=gpu:1                    
#SBATCH --cpus-per-task=128
#SBATCH --time=1:00:00                 
#SBATCH --output=/scratch/s02210660/diploma-llm/Qwen2.5-3B-2nd_prune/runs/model_prune_%j.out     
#SBATCH --error=/scratch/s02210660/diploma-llm/Qwen2.5-3B-2nd_prune/runs/model_prune_%j.err      

export NNODES=1
export GPUS_PER_NODE=1

export head_node=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus 0 -w "$head_node" hostname --ip-address)

echo Head Node: $head_node
echo Head Node IP: $head_node_ip
echo "${head_node_ip: -1}"

export PRUNE_LAYERS="32"
export MODEL_PATH=/scratch/s02210660/diploma-llm/qwen2.5-ft_2048/new_4type__qwen2.5-3b-prune-3,28,29_logits:100+hs-last-learn+last-models_lr-1e-4_bs-2_gracm-4_maxlen-2048_ds-size-500000_none/student
export SAVE_PATH="/scratch/s02210660/diploma-llm/Qwen2.5-3B-pruning-4-type/qwen2.5-3b-pruned3,28,29,X-vars/qwen2.5-3B-pruned3,28,29,32/model"

echo SAVE_PATH 


srun --container-image /scratch/s02210660/diploma-llm/ngc_cuda_pytorch_24_04_v1+latest.sqsh \
     --container-workdir /scratch/s02210660/diploma-llm \
     --container-mounts /scratch/s02210660/diploma-llm:/scratch/s02210660/diploma-llm \
     bash -c "cd /scratch/s02210660/diploma-llm && \
               ./trim_model.sh --prune_layers $PRUNE_LAYERS \
                                    --model_path $MODEL_PATH \
                                    --save_path $SAVE_PATH" 