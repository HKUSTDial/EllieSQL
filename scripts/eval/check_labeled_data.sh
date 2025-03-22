#!/bin/bash

# bash scripts/eval/check_labeled_data.sh

# File to check
LABELED_DATA_FILE="data/labeled/bird_dev_pipeline_label.jsonl"

# 只查看分布统计
python -m src.router.check_labeled_data $LABELED_DATA_FILE --distribution_only

# 只查看label为1的采样结果
python -m src.router.check_labeled_data $LABELED_DATA_FILE --label 1 --sample_size 5 --sample_only

# 同时查看分布和采样 (默认行为)
python -m src.router.check_labeled_data $LABELED_DATA_FILE --label 1 --sample_size 5