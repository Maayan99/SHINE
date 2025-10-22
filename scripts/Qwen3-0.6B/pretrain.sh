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
SOURCE=transmla
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4

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
    mode=pretrain \
    data.source=$SOURCE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.eval_batch_size=$TEST_BATCH_SIZE \
    run.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    > tmp_pretrain.txt 2>&1 &
