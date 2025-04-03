#!/bin/bash

# bash scripts/eval/check_labeled_data.sh

# File to check
LABELED_DATA_FILE="data/labeled/bird_dev_pipeline_label.jsonl"

# Only view distribution statistics
python -m src.router.check_labeled_data $LABELED_DATA_FILE --distribution_only

# Only view samples with label 1
python -m src.router.check_labeled_data $LABELED_DATA_FILE --label 1 --sample_size 5 --sample_only

# View both distribution and sampling (default)
python -m src.router.check_labeled_data $LABELED_DATA_FILE --label 1 --sample_size 5