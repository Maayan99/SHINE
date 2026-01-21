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

NAME=8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150_lr2e-5
NUM_GPUS=8
MASTER_PORT=18900             
CONFIG_NAME="Qwen3-8B"
NUM_EPOCHS=1
EVAL_STEPS=625
SAVE_STEPS=625
GRADIENT_ACCUMULATION_STEPS=4
USE_GRADIENT_CHECKPOINT=False
CONTEXT_MAX_LEN=2380
CONVERSATION_MAX_LEN=100
RESUME_GLOBAL_STEP=latest
SOURCE=ift-c1qa
WARMUP_STEPS=400
LEARNING_RATE=2e-5
TYPE=transformer
NUM_LAYERS=4
METHOD=rl
LORA_R=8
METALORA_R=128

# Find available port
while true; do
    if ! nc -z 127.0.0.1 $MASTER_PORT; then
        break
    fi
    MASTER_PORT=$((MASTER_PORT + 1))
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
    run.use_gradient_checkpoint=$USE_GRADIENT_CHECKPOINT \
    optim.num_epochs=$NUM_EPOCHS \
    eval.eval_steps=$EVAL_STEPS \
    save.save_steps=$SAVE_STEPS \
    data.context_max_length=$CONTEXT_MAX_LEN \
    data.conversation_max_length=$CONVERSATION_MAX_LEN \
    run.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    resume_global_step=$RESUME_GLOBAL_STEP \
    data.source=$SOURCE \
    optim.warmup_steps=$WARMUP_STEPS \
    optim.learning_rate=$LEARNING_RATE \
    metanetwork.type=$TYPE \
    metanetwork.transformer_cfg.num_layers=$NUM_LAYERS \
    metanetwork.method=$METHOD \
    model.lora_r=$LORA_R \
    model.metalora_r=$METALORA_R \
    > tmp_metatrain_$NAME.txt 2>&1 &
