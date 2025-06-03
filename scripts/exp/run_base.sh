#!/bin/bash

# bash scripts/exp/run_base.sh

# Set environment variables
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=7

# Basic parameters
MAX_WORKERS=64

# Run different datasets:
# - Spider dev: ./data/formatted_spider_dev.json
# - Spider realistic: ./data/formatted_spider_realistic.json
# - Spider syn: ./data/formatted_spider_syn.json
# - Bird dev: ./data/formatted_bird_dev.json
TEST_FILE="./data/formatted_bird_dev.json"

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

