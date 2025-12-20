#!/bin/bash

NAME=8gpu_4lora_4metalora_lr5e-5_grouppretrain_6layer_1270
MASTER_PORT=18920       
CONFIG_NAME="Qwen3-8B"       
SOURCE=grouptransmla
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
USE_GRADIENT_CHECKPOINT=False
RESUME_GLOBAL_STEP=latest   # -1: don't resume,   int: resume from global steps,  latest: resume from latest
LEARNING_RATE=5e-5
CONVERSATION_MAX_LEN=1270   # Extra base len: 0 Extra chat len per turn: 11
CONTEXT_MAX_LEN=$((CONVERSATION_MAX_LEN - 9)) # $((CONVERSATION_MAX_LEN - 10))
TYPE=transformer
NUM_LAYERS=6
WARMUP_STEPS=200
METHOD=rl
visualize_steps=1
visualize_mode=pretrain

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

python \
    generate_loradict_statistics.py \
    --config-name $CONFIG_NAME \
    name=$NAME \
    mode=visualize \
    visualize.visualize_mode=$visualize_mode \
    data.source=$SOURCE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.eval_batch_size=$TEST_BATCH_SIZE \
    run.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    run.use_gradient_checkpoint=$USE_GRADIENT_CHECKPOINT \
    resume_global_step=$RESUME_GLOBAL_STEP \
    optim.learning_rate=$LEARNING_RATE \
    metanetwork.type=$TYPE \
    data.conversation_max_length=$CONVERSATION_MAX_LEN \
    data.context_max_length=$CONTEXT_MAX_LEN \
    metanetwork.transformer_cfg.num_layers=$NUM_LAYERS \
    optim.warmup_steps=$WARMUP_STEPS \
    metanetwork.method=$METHOD \
    visualize.visualize_steps=$visualize_steps \
    > tmp_loradictstatistics_$NAME.txt 2>&1
