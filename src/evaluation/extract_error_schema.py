import json
import argparse

if __name__ == "__main__":
    """
    Extract linked schema results for the objects whose generation results are wrong.  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_sql_path', type=str, required=True,
                       help='the error_sql_results.jsonl file')
    parser.add_argument('--linked_schema_path', type=str, required=True,
                       help='the linked_schema_results.jsonl file')
    parser.add_argument('--error_schema_results', type=str, required=True,
                       help='the output path for generated error_schema_results.jsonl')
    args = parser.parse_args()

    error_sql_path=args.error_sql_path
    linked_schema_path=args.linked_schema_path
    error_schema_results=args.error_schema_results

    # # Path to the files
    # error_sql_path = "results/raw_results/qwen_classifier_sft/error_sql_results.jsonl"       # error sql path
    # linked_schema_path = "results/raw_results/qwen_classifier_sft/linked_schema_results.jsonl"     # linked schema path
    # error_schema_results = "results/raw_results/qwen_classifier_sft/error_schema_results.jsonl"      # output file path (file3)

    # Read the file2, store all objects in a dictionary, and use query_id as the key for easy lookup
    query_mapping = {}
    with open(linked_schema_path, "r", encoding="utf-8") as f2:
        for line in f2:
            if line.strip():  # Exclude empty lines
                obj = json.loads(line)
                query_id = obj.get("query_id")
                if query_id is not None:
                    query_mapping[query_id] = obj

    # Iterate over file1, find the corresponding object in file2, and write to the output file
    with open(error_sql_path, "r", encoding="utf-8") as f1, open(error_schema_results, "w", encoding="utf-8") as fout:
        for line in f1:
            if line.strip():
                obj1 = json.loads(line)
                question_id = obj1.get("question_id")
                if question_id in query_mapping:
                    matched_obj = query_mapping[question_id]
                    fout.write(json.dumps(matched_obj, ensure_ascii=False) + "\n")
