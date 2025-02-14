#!/bin/bash

# Specify the labeled source dataset (for SFT dataset preparation) path
SFT_DATASET="bird_train_full_roberta"
# SFT_DATASET="bird_dev_full"
# Specify the SFT dataset directory
LABELED_FILE="data/labeled/bird_train_pipeline_label.jsonl"
# LABELED_FILE="data/labeled/bird_dev_pipeline_label.jsonl"
# Specify the SFT config file
SFT_CONFIG="roberta_config"  # 使用 config/roberta_config.yaml

# 运行数据处理脚本
python -m src.sft.prepare_classifier_sft_data \
    --sft_dataset ${SFT_DATASET} \
    --labeled_file ${LABELED_FILE}

# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 在多卡RTX 4090上分布式训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=29500 \
    -m src.sft.roberta_classifier_sft \
    --sft_config ${SFT_CONFIG} \
    --sft_dataset ${SFT_DATASET}

# 推理示例
echo -e "\n\n----------------- Inference example of RoBERTa Classifier SFT -----------------\n"
python -m src.sft.roberta_classifier_inference

# tensorboard可视化
# tensorboard --logdir logs/sft/roberta_classifier 