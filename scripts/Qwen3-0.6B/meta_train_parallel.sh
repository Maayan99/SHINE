#!/bin/bash

#SBATCH -J metalora
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o metalora.out
#SBATCH -e metalora.err


NUM_GPUS=8
MASTER_PORT=18900             
CONFIG_NAME="Qwen3-0.6B"       
NUM_EPOCHS=3
EVAL_STEPS=1250
SAVE_STEPS=1250
USE_GRADIENT_CHECKPOINT=False
MAX_LEN=1152
RESUME_GLOBAL_STEP=latest

# Find available port
while true; do
    if ! nc -z 127.0.0.1 $MASTER_PORT; then
        break
    fi
    ((MASTER_PORT++))
done

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=INFO

nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    meta_train_parallel.py \
    --config-name $CONFIG_NAME \
    run.use_gradient_checkpoint=$USE_GRADIENT_CHECKPOINT \
    optim.num_epochs=$NUM_EPOCHS \
    eval.eval_steps=$EVAL_STEPS \
    save.save_steps=$SAVE_STEPS \
    data.max_length=$MAX_LEN \
    resume_global_step=$RESUME_GLOBAL_STEP \
    > tmp_metatrain.txt 2>&1 &
