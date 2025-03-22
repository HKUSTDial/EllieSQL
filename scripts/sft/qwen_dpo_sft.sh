#!/bin/bash

# bash scripts/sft/qwen_dpo_sft.sh


# Specify the labeled source dataset (for SFT dataset preparation) path
SFT_DATASET="bird_train_full_qwen"
# SFT_DATASET="bird_dev_full"
# Specify the SFT dataset directory
LABELED_FILE="data/labeled/bird_train_pipeline_label.jsonl"
# LABELED_FILE="data/labeled/bird_dev_pipeline_label.jsonl"
# Specify the SFT config file
SFT_CONFIG="sft_config"  # 使用 config/sft_config.yaml


python -m src.dpo.prepare_dpo_data
python -m src.dpo.generate_dpo_pairs
python -m src.dpo.qwen_dpo_train