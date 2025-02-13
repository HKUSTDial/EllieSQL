python -m examples.main
python -m examples.run_scripts
python -m src.run
python -m src.router.labeler

python -m src.evaluation.compute_EX
python -m src.evaluation.compute_EX2
python -m src.evaluation.count_empty_goldsql_result
python -m src.evaluation.delete_empty_item
python -m src.evaluation.evaluate_schema_linking
python -m src.evaluation.analyze_router

python -m src.classifier.basic_classifier