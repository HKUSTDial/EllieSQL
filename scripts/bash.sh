python -m examples.main
python -m examples.main_local
python -m examples.run_scripts
python -m src.run
python -m src.router.labeler

python -m src.evaluation.compute_ex
python -m src.evaluation.count_empty_goldsql_result
python -m src.evaluation.delete_empty_item
python -m src.evaluation.evaluate_schema_linking
python -m src.evaluation.analyze_router


# for error analysis (from error_analysis main to find readme)
python -m src.evaluation.extract_error_results
python -m src.evaluation.extract_error_schema
python -m src.evaluation.error_analysis


python -m src.classifier.basic_classifier


python -m src.dpo.prepare_dpo_data
python -m src.dpo.generate_dpo_pairs
python -m src.dpo.qwen_dpo_train
