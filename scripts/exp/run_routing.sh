#!/bin/bash

# bash scripts/exp/run_routing.sh

# Set environment variables
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=7

# Basic parameters
MAX_WORKERS=64
BACKBONE="gpt-3.5-turbo"
TRAIN_FILE="./data/labeled/bird_train_pipeline_label.jsonl"

# Run different datasets:
# - Spider dev: ./data/formatted_spider_dev.json
# - Spider realistic: ./data/formatted_spider_realistic.json
# - Spider syn: ./data/formatted_spider_syn.json
# - Spider test: ./data/formatted_spider_test.json
# - Bird dev: ./data/formatted_bird_dev.json
TEST_FILE="./data/formatted_bird_dev.json"

# No.1: KNN Router
echo "Running experiment with KNN Router..."
python -m src.run_routing \
    --router knn \
    --test_file $TEST_FILE \
    --train_file_path $TRAIN_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE


# No.2: Qwen Classifier Router
echo "Running experiment with Qwen Classifier Router..."
python -m src.run_routing \
    --router qwen \
    --router_path "/data/ElephantSQL_Checkpoints/qwen_classification_sft/final_model_classifier" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE


# No.3: RoBERTa Classifier Router
echo "Running experiment with RoBERTa Classifier Router..."
python -m src.run_routing \
    --router roberta \
    --router_path "/data/ElephantSQL_Checkpoints/roberta_classification_sft/final_model_roberta_lora" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE


# No.4: Qwen Cascade Router (sub-classifier checkpoints are in cascade folder)
echo "Running experiment with Qwen Cascade Router..."
python -m src.run_routing \
    --router qwen_cascade \
    --router_path "/data/ElephantSQL_Checkpoints/qwen_cascade_sft" \
    --confidence_threshold 0.6 \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE


# No.5: RoBERTa Cascade Router
echo "Running experiment with RoBERTa Cascade Router..."
python -m src.run_routing \
    --router roberta_cascade \
    --router_path "/data/ElephantSQL_Checkpoints/roberta_cascade_sft" \
    --confidence_threshold 0.6 \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE


# No.6: Qwen Pairwise Router
echo "Running experiment with Qwen Pairwise Router..."
python -m src.run_routing \
    --router qwen_pairwise \
    --router_path "/data/ElephantSQL_Checkpoints/qwen_pairwise_sft/final_model_classifier" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE


# No.7: Qwen DPO Router
echo "Running experiment with Qwen DPO Router..."
python -m src.run_routing \
    --router qwen_dpo \
    --router_path "/data/ElephantSQL_Checkpoints/qwen_dpo" \
    --test_file $TEST_FILE \
    --max_workers $MAX_WORKERS \
    --backbone_model $BACKBONE

