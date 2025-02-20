#!/bin/bash

# bash scripts/qwen_pairwise_sft.sh

# For test during development
# PAIRWISE_DATASET="bird_train_test"
# LABELED_FILE="data/labeled/sampled_bird_demo_pipeline_preference.jsonl"

# For real training
PAIRWISE_DATASET="bird_train_pairwise"
LABELED_FILE="data/labeled/bird_train_pipeline_label.jsonl"

# 指定配置文件
PAIRWISE_CONFIG="pairwise_config"  # 使用 config/pairwise_config.yaml

# 首先运行数据处理脚本准备pairwise数据
python -m src.sft.prepare_pairwise_data \
    --pairwise_dataset ${PAIRWISE_DATASET} \
    --labeled_file ${LABELED_FILE}

# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 在多卡上分布式训练
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=29500 \
    -m src.sft.qwen_pairwise_sft \
    --pairwise_config ${PAIRWISE_CONFIG} \
    --pairwise_dataset ${PAIRWISE_DATASET}

# 可选：运行推理示例
# echo -e "\n\n----------------- Inference example of Qwen Pairwise SFT -----------------\n"
python -m src.sft.qwen_pairwise_inference

# tensorboard可视化
# tensorboard --logdir logs/sft/qwen_classifier 