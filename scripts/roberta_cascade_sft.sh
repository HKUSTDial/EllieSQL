#!/bin/bash

# bash scripts/roberta_cascade_sft.sh

# Specify the labeled source dataset (for SFT dataset preparation) path
CASCADE_DATASET="bird_train_cascade"
# Specify the SFT dataset directory
LABELED_FILE="data/labeled/bird_train_pipeline_label.jsonl"
# Specify the SFT config file
CASCADE_CONFIG="roberta_config"

# 运行数据处理脚本
python -m src.sft.prepare_cascade_data \
    --cascade_dataset ${CASCADE_DATASET} \
    --labeled_file ${LABELED_FILE} \
    --basic_balance false \
    --intermediate_balance true \
    --advanced_balance true


# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 在多卡RTX 4090上分布式训练
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 训练basic pipeline的分类器
torchrun --nproc_per_node=4 --master_port=29500 \
    -m src.sft.roberta_cascade_sft \
    --cascade_dataset ${CASCADE_DATASET} \
    --pipeline_type basic \
    --model_name basic_classifier \
    --sft_config ${CASCADE_CONFIG}

# 推理示例
echo -e "\n\n----------------- Inference example of RoBERTa Cascade SFT (basic pipeline) -----------------\n"
python -m src.sft.roberta_cascade_inference \
    --pipeline_type basic \
    --confidence_threshold 0.5


# 训练intermediate pipeline的分类器
torchrun --nproc_per_node=4 --master_port=29500 \
    -m src.sft.roberta_cascade_sft \
    --cascade_dataset ${CASCADE_DATASET} \
    --pipeline_type intermediate \
    --model_name intermediate_classifier \
    --sft_config ${CASCADE_CONFIG}

# 推理示例
echo -e "\n\n----------------- Inference example of RoBERTa Cascade SFT (intermediate pipeline) -----------------\n"
python -m src.sft.roberta_cascade_inference \
    --pipeline_type intermediate \
    --confidence_threshold 0.5