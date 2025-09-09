#!/bin/bash
# scripts/launch.sh

# Launch script for AGRPO training with DeepSpeed on multi-GPU setup
# Optimized for Kaggle's dual T4 GPUs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=/kaggle/working/cache
export HF_HOME=/kaggle/working/cache

# Create necessary directories
mkdir -p outputs/agrpo_deepspeed
mkdir -p logs

# Number of GPUs
NUM_GPUS=2

# Training arguments
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR="outputs/agrpo_deepspeed"
BATCH_SIZE=1
GRAD_ACCUM=4
EPOCHS=2
LR=2e-4
SAMPLE_SIZE=1000

# For Kaggle environment with DeepSpeed
echo "Starting AGRPO training with DeepSpeed ZeRO-3..."
echo "Using $NUM_GPUS GPUs"

# Option 1: Using accelerate launch (recommended)
accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file=config/deepspeed_config.json \
    scripts/train.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $EPOCHS \
    --learning_rate $LR \
    --sample_size $SAMPLE_SIZE \
    --fp16 \
    --verbose \
    --checkpoint_every_n_steps 10

# Option 2: Using deepspeed directly (alternative)
# deepspeed --num_gpus=$NUM_GPUS \
#     scripts/train.py \
#     --model_name $MODEL_NAME \
#     --output_dir $OUTPUT_DIR \
#     --batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps $GRAD_ACCUM \
#     --num_epochs $EPOCHS \
#     --learning_rate $LR \
#     --sample_size $SAMPLE_SIZE \
#     --fp16 \
#     --verbose \
#     --deepspeed_config config/deepspeed_config.json

# Option 3: Using torchrun (PyTorch native)
# torchrun --nproc_per_node=$NUM_GPUS \
#     scripts/train.py \
#     --model_name $MODEL_NAME \
#     --output_dir $OUTPUT_DIR \
#     --batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps $GRAD_ACCUM \
#     --num_epochs $EPOCHS \
#     --learning_rate $LR \
#     --sample_size $SAMPLE_SIZE \
#     --fp16 \
#     --verbose

echo "Training completed!"