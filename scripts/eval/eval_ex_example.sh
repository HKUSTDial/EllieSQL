#!/bin/bash

# bash scripts/eval/eval_ex_example.sh

GOLD_SQL_FILE="./data/formatted_bird_dev.json"

# If no result path parameter is provided, test for the latest result in results/intermediate_results
python -m src.evaluation.compute_ex \
    --gold_sql_file $GOLD_SQL_FILE

# If a result path parameter is provided, test for the specific result in the given path
python -m src.evaluation.compute_ex \
    --gold_sql_file $GOLD_SQL_FILE \
    --result_path ./results/intermediate_results/20240315_123456
