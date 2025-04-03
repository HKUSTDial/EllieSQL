#!/bin/bash

# bash scripts/sft/qwen_pairwise_sft.sh

PAIRWISE_DATASET="bird_train_pairwise"
LABELED_FILE="data/labeled/bird_train_pipeline_label.jsonl"
PAIRWISE_CONFIG="pairwise_config"

# Run data processing script to prepare pairwise data
python -m src.sft.prepare_pairwise_data \
    --pairwise_dataset ${PAIRWISE_DATASET} \
    --labeled_file ${LABELED_FILE}

# Set environment variables
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Train on multi-card in parallel
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=29500 \
    -m src.sft.qwen_pairwise_sft \
    --pairwise_config ${PAIRWISE_CONFIG} \
    --pairwise_dataset ${PAIRWISE_DATASET}

# Optional: Run inference example
# echo -e "\n\n----------------- Inference example of Qwen Pairwise SFT -----------------\n"
python -m src.sft.qwen_pairwise_inference

# Tensorboard visualization
# tensorboard --logdir logs/sft/qwen_classifier 