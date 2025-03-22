#!/bin/bash

# bash scripts/exp/run_base.sh

# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=7

# 基础参数
TEST_FILE="./data/formatted_bird_dev.json"
MAX_WORKERS=64

# GB pipeline
echo "Running GB pipeline..."
python -m src.run_base \
    --pipeline GB \
    --schema_model "gpt-3.5-turbo" \
    --generator_model "gpt-3.5-turbo" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS


# GM pipeline
echo "Running GM pipeline..."
python -m src.run_base \
    --pipeline GM \
    --schema_model "gpt-3.5-turbo" \
    --generator_model "gpt-3.5-turbo" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS


# GA pipeline
echo "Running GA pipeline..."
python -m src.run_base \
    --pipeline GA \
    --schema_model "gpt-3.5-turbo" \
    --generator_model "gpt-3.5-turbo" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS

