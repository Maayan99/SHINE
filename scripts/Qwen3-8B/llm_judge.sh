#!/bin/bash
#SBATCH -J llmjudge     # Job name
#SBATCH -p IAI_SLURM_HGX     # Partition / queue name
#SBATCH --qos=16gpu-hgx      # QOS (adjust as needed)
#SBATCH -N 1                 # Number of nodes
#SBATCH --gres=gpu:0        # GPUs per node
#SBATCH --time=48:00:00     # Max runtime (48h)
#SBATCH -c 64               # CPU cores
#SBATCH -o llmjudge.out         # Stdout file
#SBATCH -e llmjudge.err         # Stderr file

# === User-configurable variables ===
NAME= ????????
MASTER_PORT=18900
CONFIG_NAME="Qwen3-8B"           # Hydra config name (e.g., base.yaml)

# === Find a free port (in case default is in use) ===
while true; do
    if ! nc -z 127.0.0.1 $MASTER_PORT; then
        break
    fi
    MASTER_PORT=$((MASTER_PORT + 1))
done

# === Environment variables (recommended for PyTorch & Hydra) ===
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4

# === Run distributed job ===
python \
    llm_judge.py \
    --config-name $CONFIG_NAME \
    > tmp_llmjudge_$NAME.txt 2>&1
