#!/bin/bash

#SBATCH -J metalora
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o metalora.out
#SBATCH -e metalora.err

NAME=tmp3
NUM_GPUS=2
MASTER_PORT=18930      
CONFIG_NAME="Qwen3-8B"       
SOURCE=transmla
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
USE_GRADIENT_CHECKPOINT=False
RESUME_GLOBAL_STEP=latest   # -1: don't resume,   int: resume from global steps,  latest: resume from latest
LEARNING_RATE=5e-5
TYPE=transformer
CONVERSATION_MAX_LEN=1440
CONTEXT_MAX_LEN=1431
NUM_LAYERS=4
WARMUP_STEPS=200
METHOD=rl
LORA_R=8
METALORA_R=8

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
    name=$NAME \
    mode=pretrain \
    data.source=$SOURCE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.eval_batch_size=$TEST_BATCH_SIZE \
    run.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    run.use_gradient_checkpoint=$USE_GRADIENT_CHECKPOINT \
    resume_global_step=$RESUME_GLOBAL_STEP \
    optim.learning_rate=$LEARNING_RATE \
    data.conversation_max_length=$CONVERSATION_MAX_LEN \
    data.context_max_length=$CONTEXT_MAX_LEN \
    metanetwork.type=$TYPE \
    metanetwork.transformer_cfg.num_layers=$NUM_LAYERS \
    optim.warmup_steps=$WARMUP_STEPS \
    metanetwork.method=$METHOD \
    model.lora_r=$LORA_R \
    model.metalora_r=$METALORA_R \
    > tmp_pretrain_$NAME.txt 2>&1 &
