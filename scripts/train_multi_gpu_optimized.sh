#!/bin/bash

# Optimized Multi-GPU Training Script for Sign Language Translation
# Uses PyTorch DistributedDataParallel (DDP) with performance optimizations
# Usage: CUDA_VISIBLE_DEVICES=0,1,3 bash scripts/train_multi_gpu_optimized.sh configs/t5_base_isign_optimized.yaml 3

set -e  # Exit on error

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/train_multi_gpu_optimized.sh <config_file> [num_gpus]"
    echo "Example: CUDA_VISIBLE_DEVICES=0,1,3 bash scripts/train_multi_gpu_optimized.sh configs/t5_base_isign_optimized.yaml 3"
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
echo "Starting Optimized Multi-GPU Training"
echo "Config: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Make src/ visible to all DDP processes
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
