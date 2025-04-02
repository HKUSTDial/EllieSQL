#!/bin/bash

# bash scripts/sft/qwen_dpo_sft.sh


# Specify the labeled source dataset (for SFT dataset preparation) path
SFT_DATASET="bird_train_dataset"
# Specify the SFT dataset directory
LABELED_FILE="data/labeled/bird_train_pipeline_label.jsonl"

DPO_DATA_FILE="data/dpo/bird_train_dataset/classifier_train_pre.json"

PREFERENCE_FILE="data/dpo/bird_train_dataset/classifier_train.json" 
# Specify the DPO config file
DPO_CONFIG="dpo_config"


# 运行数据处理脚本
python -m src.dpo.prepare_dpo_data \
    --sft_dataset ${SFT_DATASET} \
    --labeled_file ${LABELED_FILE}


python -m src.dpo.generate_dpo_pairs \
    --labeled_file ${DPO_DATA_FILE} \
    --preference_file ${PREFERENCE_FILE}



# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 在多卡RTX 4090上分布式训练
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m src.dpo.qwen_dpo_train \
    --dpo_config ${DPO_CONFIG}