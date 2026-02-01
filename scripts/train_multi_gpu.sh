#!/bin/bash

# Multi-GPU Training Script for Sign Language Translation
# Uses PyTorch DistributedDataParallel (DDP)
# Usage: bash scripts/train_multi_gpu.sh <config_file> <num_gpus>

set -e  # Exit on error

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/train_multi_gpu.sh <config_file> [num_gpus]"
    echo "Example: bash scripts/train_multi_gpu.sh configs/t5_base_isign.yaml 4"
    exit 1
fi

CONFIG_FILE=$1
NUM_GPUS=${2:-4}

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "=========================================="
echo "Starting Multi-GPU Training"
echo "Config: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# 🔑 CRITICAL: make src/ visible to all DDP processes
export PYTHONPATH="$(pwd)"

# Launch training
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py --config "$CONFIG_FILE"

echo "=========================================="
echo "Training Completed!"
echo "=========================================="
