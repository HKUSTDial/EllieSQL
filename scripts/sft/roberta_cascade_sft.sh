#!/bin/bash

# bash scripts/sft/roberta_cascade_sft.sh

# Specify the labeled source dataset (for SFT dataset preparation) path
CASCADE_DATASET="bird_train_cascade"

# Specify the SFT dataset directory
LABELED_FILE="data/labeled/bird_train_pipeline_label.jsonl"

# Specify the SFT config file
CASCADE_CONFIG="roberta_cascade_config"

# Run data processing script
python -m src.sft.prepare_cascade_data \
    --cascade_dataset ${CASCADE_DATASET} \
    --labeled_file ${LABELED_FILE} \
    --basic_balance false \
    --intermediate_balance true \
    --advanced_balance true


# Set environment variables
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Train on multi-card RTX 4090 in parallel
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Train basic pipeline classifier
echo -e "\n\n----------------- Training RoBERTa Cascade SFT (basic pipeline) -----------------\n"
torchrun --nproc_per_node=4 --master_port=29500 \
    -m src.sft.roberta_cascade_sft \
    --cascade_dataset ${CASCADE_DATASET} \
    --pipeline_type basic \
    --model_name basic_classifier \
    --sft_config ${CASCADE_CONFIG}

# Inference example
echo -e "\n\n----------------- Inference example of RoBERTa Cascade SFT (basic pipeline) -----------------\n"
python -m src.sft.roberta_cascade_inference \
    --pipeline_type basic \
    --confidence_threshold 0.5


# Train intermediate pipeline classifier
echo -e "\n\n----------------- Training RoBERTa Cascade SFT (intermediate pipeline) -----------------\n"
torchrun --nproc_per_node=4 --master_port=29500 \
    -m src.sft.roberta_cascade_sft \
    --cascade_dataset ${CASCADE_DATASET} \
    --pipeline_type intermediate \
    --model_name intermediate_classifier \
    --sft_config ${CASCADE_CONFIG}

# Inference example
echo -e "\n\n----------------- Inference example of RoBERTa Cascade SFT (intermediate pipeline) -----------------\n"
python -m src.sft.roberta_cascade_inference \
    --pipeline_type intermediate \
    --confidence_threshold 0.5