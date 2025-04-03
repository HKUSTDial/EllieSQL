import os

from ..core.utils import load_json, load_jsonl
from ..core.sql_execute import *
from .compute_ex import compare_sql_results
from ..core.config import Config


def analyze_results(direct_selector_path, sampled_merged_path):
    # Read the sampled_merged.json file
    sampled_merged = load_json(sampled_merged_path)
    selector_data = load_jsonl(direct_selector_path)

    # Initialize the statistics variables
    total_objects = 0
    both_correct = 0
    one_correct = 0
    both_wrong = 0
    only_one_candidate_and_correct = 0
    only_one_candidate_and_wrong = 0

    # Read the DirectSelector.jsonl file
    for obj in selector_data:
        query_id = obj['query_id']
        candidate_sqls = obj['input']['candidate_sqls']
        for item in sampled_merged:
            curr_q = item.get("question_id")
            if(curr_q == query_id):
                total_objects += 1

                source = item.get("source", "")
                difficulty = item.get("difficulty")
                db_id = item.get("db_id", "")
                db_folder = f"{source}_{db_id}"
                db_file = f"{db_id}.sqlite"
                db_path = str(Config().database_dir / db_folder / db_file)
                gold_sql = item.get("gold_SQL")
                # Compare the execution results of the two candidate_sqls with gold_SQL
                result1 = compare_sql_results(db_path, gold_sql, candidate_sqls[0])
                if(len(candidate_sqls) == 1):
                    if(result1):
                        only_one_candidate_and_correct += 1
                    else:
                        only_one_candidate_and_wrong += 1
                    break
                result2 = compare_sql_results(db_path, gold_sql, candidate_sqls[1])

                if result1 and result2:
                    both_correct += 1
                elif result1 or result2:
                    one_correct += 1
                else:
                    both_wrong += 1
                break
            

    # Calculate the ratios
    both_correct_ratio = both_correct / total_objects
    one_correct_ratio = one_correct / total_objects
    both_wrong_ratio = both_wrong / total_objects
    only_one_candidate_and_correct_ratio = only_one_candidate_and_correct / total_objects
    only_one_candidate_and_wrong_ratio = only_one_candidate_and_wrong / total_objects
    
    # Print the results
    print(f"The total number of objects: {total_objects}")
    print(f"The number of objects where both SQL execution results are the same as gold_SQL: {both_correct} (Percentage: {both_correct_ratio:.2%})")
    print(f"The number of objects where only one SQL execution result is the same as gold_SQL: {one_correct} (Percentage: {one_correct_ratio:.2%})")
    print(f"The number of objects where both SQL execution results are wrong: {both_wrong} (Percentage: {both_wrong_ratio:.2%})")
    print(f"The number of objects where only one candidate SQL is correct: {only_one_candidate_and_correct} (Percentage: {only_one_candidate_and_correct_ratio:.2%})")
    print(f"The number of objects where only one candidate SQL is wrong: {only_one_candidate_and_wrong} (Percentage: {only_one_candidate_and_wrong_ratio:.2%})")

if __name__ == "__main__":
    direct_selector_path = "./results\important_results\\20250114_003759\DirectSelector.jsonl"
    #merge_dev_demo_file = "./data/merge_dev_demo.json"
    sampled_merged_path = "./data/sampled_merged.json"

    analyze_results(direct_selector_path, sampled_merged_path)
    exit()
