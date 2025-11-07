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


NUM_GPUS=4
MASTER_PORT=18900             
CONFIG_NAME="Qwen3-8B"       
TEST_BATCH_SIZE=8
TEST_GLOBAL_STEP=latest
        

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
    test.py \
    --config-name $CONFIG_NAME \
    test.batch_size=$TEST_BATCH_SIZE \
    test_global_step=$TEST_GLOBAL_STEP \
    > tmp_test.txt 2>&1 &
