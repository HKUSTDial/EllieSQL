#!/bin/bash

# bash scripts/exp/run_routing.sh

# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=7

# 基础参数
TRAIN_FILE="./data/labeled/bird_train_pipeline_label.jsonl"
TEST_FILE="./data/formatted_bird_dev.json"
MAX_WORKERS=64
BACKBONE="gpt-3.5-turbo"

# 实验1: KNN Router
echo "Running experiment with KNN Router..."
python -m src.run_routing \
    --router knn \
    --test_file $TEST_FILE \
    --train_file_path $TRAIN_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE

# 实验2: Qwen Classifier Router
echo "Running experiment with Qwen Classifier Router..."
python -m src.run_routing \
    --router qwen \
    --router_path "/path/to/saves/final_model_qwen_lora" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE

# 实验3: RoBERTa Classifier Router
echo "Running experiment with RoBERTa Classifier Router..."
python -m src.run_routing \
    --router roberta \
    --router_path "/path/to/saves/final_model_roberta_lora" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE

# 实验4: Qwen Cascade Router (sub-classifier checkpoints are in cascade folder)
echo "Running experiment with Qwen Cascade Router..."
python -m src.run_routing \
    --router qwen_cascade \
    --router_path "/path/to/qwen/cascade" \
    --confidence_threshold 0.6 \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE

# 实验5: RoBERTa Cascade Router
echo "Running experiment with RoBERTa Cascade Router..."
python -m src.run_routing \
    --router roberta_cascade \
    --router_path "/path/to/roberta/cascade" \
    --confidence_threshold 0.6 \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE

# 实验6: Qwen Pairwise Router
echo "Running experiment with Qwen Pairwise Router..."
python -m src.run_routing \
    --router qwen_pairwise \
    --router_path "/path/to/qwen/pairwise/final_model_classifier" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE

# 实验7: Qwen DPO Router
echo "Running experiment with Qwen DPO Router..."
python -m src.run_routing \
    --router qwen_dpo \
    --router_path "/path/to/qwen/dpo/qwen_dpo" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE

