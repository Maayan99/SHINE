#!/bin/bash

#SBATCH -J test
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o test.out
#SBATCH -e test.err

NAME=8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150
NUM_GPUS=8
MASTER_PORT=18900             
CONFIG_NAME="Qwen3-8B"       
TEST_BATCH_SIZE=4
TEST_GLOBAL_STEP=epoch-1
TEST_SOURCE=wikitext
MAX_NEW_TOKEN=2500
NUM_LAYERS=4
METHOD=rl
LORA_R=8
METALORA_R=128
        

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
    test_pretrain.py \
    --config-name $CONFIG_NAME \
    name=$NAME \
    test.batch_size=$TEST_BATCH_SIZE \
    test_global_step=$TEST_GLOBAL_STEP \
    test.source=$TEST_SOURCE \
    metanetwork.transformer_cfg.num_layers=$NUM_LAYERS \
    metanetwork.method=$METHOD \
    test.max_new_tokens=$MAX_NEW_TOKENS \
    model.lora_r=$LORA_R \
    model.metalora_r=$METALORA_R \
    > tmp_test_pretrain_$NAME.txt 2>&1 &
