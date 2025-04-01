#!/bin/bash

# this file is an output file
ERROR_SQL_PATH= "results/raw_results/qwen_classifier_sft/error_sql_results.jsonl"
MERGE_DEV_DEMO_FILE= "./data/formatted_bird_dev.json"
GENERATED_SQL_PATH= "results/raw_results/qwen_classifier_sft/generated_sql_results.jsonl"


LINKED_SCHEMA_PATH= "results/raw_results/qwen_classifier_sft/linked_schema_results.jsonl"    
ERROR_SCHEMA_RESULTS_PATH= "results/raw_results/qwen_classifier_sft/error_schema_results.jsonl"      


echo -e "\n=== 1. extract error results ==="

python -m src.evaluation.extract_error_results \
  --merge_dev_demo_file $MERGE_DEV_DEMO_FILE \
  --result_path $GENERATED_SQL_PATH \
  --output_path $ERROR_SQL_PATH


echo -e "\n=== 2. extract error schema ==="

python -m src.evaluation.extract_error_schema \
  --error_sql_path $ERROR_SQL_PATH \
  --linked_schema_path $LINKED_SCHEMA_PATH \
  --error_schema_results $ERROR_SCHEMA_RESULTS_PATH 

echo -e "\n=== 3. error analysis ==="

python -m src.evaluation.error_analysis \
  --error_sql_path $ERROR_SQL_PATH \
  --error_schema_results $ERROR_SCHEMA_RESULTS_PATH \
  --merge_dev_demo_file $MERGE_DEV_DEMO_FILE
