#!/bin/bash
#SBATCH --job-name=qwen_distillation    
#SBATCH --nodes=4         
#SBATCH --gres=gpu:8                    
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00                 
#SBATCH --output=/scratch/s02210660/diploma-llm/distillation-runs/distil_qwen_%j.out     
#SBATCH --error=/scratch/s02210660/diploma-llm/distillation-runs/distil_qwen_%j.err      

export NNODES=4
export GPUS_PER_NODE=8

export head_node=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus 0 -w "$head_node" hostname --ip-address)

echo Head Node: $head_node
echo Head Node IP: $head_node_ip
echo "${head_node_ip: -1}"

#export TEACHER="/scratch/s02210660/diploma-llm/Qwen-pruned/qwen2.5-3B-prune3,28/model"


#export STUDENT="/scratch/s02210660/diploma-llm/Qwen2.5-3B-2nd_prune/qwen2.5-3B-prune20/model"
export PRUNE_LAYERS_1="3"
export PRUNE_LAYERS_2="28"
export PRUNE_LAYERS_3="28"
export PRUNE_LAYERS_4="19"

export TRAIN_LAYERS="18 19"



export STUDENT=/scratch/s02210660/diploma-llm/Qwen2.5-3B-pruning-4-type/qwen2.5-3b-pruned3,28,29,X-vars/qwen2.5-3B-pruned3,28,29,19/model
export LR=1e-4
export EPOCHS=1
export BS=2
export GRADACM=4
export MAXLEN=2048
export DSFRAC=500000
export NORMFACT=0.1

OUTPUT_DIR="qwen2.5-ft_${MAXLEN}/new_4type__qwen2.5-3b-prune-3,28,29,19_logits:100+hs-last-learn+last-models_lr-${LR}_bs-${BS}_gracm-${GRADACM}_maxlen-${MAXLEN}_ds-size-${DSFRAC}_none"
LOGGING_DIR="qwen2.5-ft_${MAXLEN}/logs-logits"


echo "Output directory: $OUTPUT_DIR"
echo "Logging directory: $LOGGING_DIR"

srun --container-image /scratch/s02210660/diploma-llm/ngc_cuda_pytorch_24_04_v1+latest.sqsh \
     --container-workdir /scratch/s02210660/diploma-llm \
     --container-mounts /scratch/s02210660/diploma-llm:/scratch/s02210660/diploma-llm \
     bash -c "cd /scratch/s02210660/diploma-llm && chmod +x run_distillation.sh && \
               ./run_distillation.sh \
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
                                    --logging_dir $LOGGING_DIR "


&& ./eval.sh