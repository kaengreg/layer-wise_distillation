#!/bin/bash
#SBATCH --job-name=BI_eval  
#SBATCH --nodes=1         
#SBATCH --gres=gpu:2                    
#SBATCH --cpus-per-task=128
#SBATCH --time=4:00:00                 
#SBATCH --output=/scratch/s02210660/diploma-llm/eval_BI_runs/eval_BI_%j.out     
#SBATCH --error=/scratch/s02210660/diploma-llm/eval_BI_runs/eval_BI_%j.err      

export NNODES=1
export GPUS_PER_NODE=2

export head_node=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus 0 -w "$head_node" hostname --ip-address)

echo Head Node: $head_node
echo Head Node IP: $head_node_ip
echo "${head_node_ip: -1}"

export DATASET=/scratch/s02210660/diploma-llm/data/val.json
export LINES=1000
export MAXLEN=2048
export GRAPH_PATH="/scratch/s02210660/diploma-llm/Qwen2.5-3B-pruning-3-type/qwen2.5-3B-pruned-21,22,19,23,24,18,17,2/qwen2.5-3B-prune-21,22,19,23,2,18,17,2+ft_BI"
export GRAPH_NAME="Importance_val_1000_BI.png"
export MODEL_PATH=/scratch/s02210660/diploma-llm/qwen2.5-ft_2048/new_3type__qwen2.5-3b-prune-21,22,19,23,24,18,17,2_logits:100+hs-last-learn+last-models_lr-1e-4_bs-2_gracm-4_maxlen-2048_ds-size-500000_none/student

srun --container-image /scratch/s02210660/diploma-llm/ngc_cuda_pytorch_24_04_v1+latest.sqsh \
     --container-workdir /scratch/s02210660/diploma-llm \
     --container-mounts /scratch/s02210660/diploma-llm:/scratch/s02210660/diploma-llm \
     bash -c "cd /scratch/s02210660/diploma-llm && \
               ./eval_BI_torchrun.sh --dataset_file $DATASET \
                                    --lines $LINES \
                                    --graph_path $GRAPH_PATH \
                                    --graph_name $GRAPH_NAME \
                                    --maxlen $MAXLEN \
                                    --model_path $MODEL_PATH \
                                    --loss_type BI"
