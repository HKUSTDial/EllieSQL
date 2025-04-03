#!/bin/bash

# bash scripts/sft/roberta_classifier_sft.sh

# Specify the labeled source dataset (for SFT dataset preparation) path
SFT_DATASET="bird_train_full_roberta"

# Specify the SFT dataset directory
LABELED_FILE="data/labeled/bird_train_pipeline_label.jsonl"

# Specify the SFT config file
SFT_CONFIG="roberta_config"

# Specify training mode (lora or full)
TRAINING_MODE="lora"

# Run data processing script
python -m src.sft.prepare_classifier_sft_data \
    --sft_dataset ${SFT_DATASET} \
    --labeled_file ${LABELED_FILE}

# Set environment variables
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Train on multi-card RTX 4090 in parallel
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=29500 \
    -m src.sft.roberta_classifier_sft \
    --sft_config ${SFT_CONFIG} \
    --sft_dataset ${SFT_DATASET} \
    --training_mode ${TRAINING_MODE}

# Inference example
echo -e "\n\n----------------- Inference example of RoBERTa Classifier SFT -----------------\n"
python -m src.sft.roberta_classifier_inference

# Tensorboard visualization
# tensorboard --logdir logs/sft/roberta_classifier 