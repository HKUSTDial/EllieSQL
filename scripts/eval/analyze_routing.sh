#!/bin/bash

# bash scripts/eval/analyze_routing.sh

# Oracle file
ORACLE_FILE="./data/labeled/bird_dev_pipeline_label.jsonl"


# Qwen Router
echo -e "\n=== Analyzing Qwen Router ==="
python -m src.evaluation.analyze_routing \
    --router_file "results/routing/qwen_classifier_sft/QwenClassifierRouter.jsonl" \
    --oracle_file "$ORACLE_FILE"

# RoBERTa Router
echo -e "\n=== Analyzing RoBERTa Router ==="
python -m src.evaluation.analyze_routing \
    --router_file "results/routing/roberta_classifier_sft/RoBERTaClassifierRouter.jsonl" \
    --oracle_file "$ORACLE_FILE"

# Qwen Cascade Router
echo -e "\n=== Analyzing Qwen Cascade Router ==="
python -m src.evaluation.analyze_routing \
    --router_file "results/routing/qwen_cascade/QwenCascadeRouter.jsonl" \
    --oracle_file "$ORACLE_FILE"

# RoBERTa Cascade Router
echo -e "\n=== Analyzing RoBERTa Cascade Router ==="
python -m src.evaluation.analyze_routing \
    --router_file "results/routing/roberta_cascade/RoBERTaCascadeRouter.jsonl" \
    --oracle_file "$ORACLE_FILE"

# Qwen Pairwise Router
echo -e "\n=== Analyzing Qwen Pairwise Router ==="
python -m src.evaluation.analyze_routing \
    --router_file "results/routing/qwen_pairwise_sft/QwenPairwiseRouter.jsonl" \
    --oracle_file "$ORACLE_FILE"

# Qwen DPO Router
echo -e "\n=== Analyzing Qwen DPO Router ==="
python -m src.evaluation.analyze_routing \
    --router_file "results/routing/qwen_dpo/DPOClassifierRouter.jsonl" \
    --oracle_file "$ORACLE_FILE"

