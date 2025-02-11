#!/bin/bash

# bash scripts/qwen_classifier_sft.sh

# Specify the labeled source dataset (for SFT dataset preparation) path
SFT_DATASET="bird_dev_test_2"
# Specify the SFT dataset directory
LABELED_FILE="data/labeled/bird_dev_pipeline_label.jsonl"
# Specify the SFT config file
SFT_CONFIG="sft_config"  # 使用 config/sft_config.yaml

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
    -m src.sft.qwen_classifier_sft \
    --sft_dataset ${SFT_DATASET} \
    --sft_config ${SFT_CONFIG}

# 推理示例
echo "----------------- Inference example of Qwen Classifier SFT -----------------"
python -m src.sft.qwen_classifier_inference

echo "----------------- Run the routing pipeline -----------------"
python -m src.run

# tensorboard可视化
# tensorboard --logdir logs/sft/qwen_classifier