import os
import argparse
from pathlib import Path
from ..core.utils import load_json, load_jsonl
from ..core.sql_execute import *
from ..core.config import Config


def find_latest_folder(directory):
    """
    Find the latest folder in intermediate results folder.
    
    :param directory: The path to the intermediate_results
    :return: The latest_folder.
    """ 
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    if not folders:
        return None  # If there is no folder, return None
    
    # Sort the folders by dictionary order, the last one is the latest
    latest_folder = sorted(folders)[-1]
    return latest_folder


def compare_sql_results(db_path, sql1, sql2, timeout=10):
    """
    Compare the ex results for two sqls.
    
    :param db_path: The path to the .sqlite file.
    :param sql1: The gold sql.
    :param sql2: The generated sql.
    :return: 1 for equal, 0 for not equal.
    """
    # Execute the first SQL statement
    result1 = execute_sql_with_timeout(db_path, sql1, timeout)
    if result1 is None:
        print("gold SQL execution failed!!!! Please check the error information.")
        return 0
    # Execute the second SQL statement
    result2 = execute_sql_with_timeout(db_path, sql2, timeout)
    if result2 is None:
        print("generated SQL did not produce any result object.")
        return 0

    if result2.result_type == SQLExecutionResultType.TIMEOUT:
        print(f"generated SQL timed out. Error information: {result2.error_message}")
        return 0
    if result2.result_type == SQLExecutionResultType.ERROR:
        print(f"generated SQL encountered an error. Error information: {result2.error_message}")
        return 0
    
    if result1.result is None:
        print("gold SQL executed successfully but the result is empty.")
        if result2.result is None:
            # If both SQLs return empty results, consider them equal
            return 1
        return 0

    if result2.result is None:
        print("generated SQL executed successfully but the result is empty.")
        return 0
    
    # Compare the results
    if set(result1.result) == set(result2.result):
        return 1
    else:
        return 0



def compute_EX_source_difficulty_based(gold_sql_file: str, result_path: str = None):
    """
    Compute the EX for the latest generated sql results, grouped by source.
    For 'bird_dev' source, further group by difficulty.
    Also compute the overall EX across all sources and difficulties.
    
    :param gold_sql_file: The path to the nl2sql file.
    :param result_path: The path to intermediate_results (upper level catalog for such as 20250103_172805).
    :return: A dictionary containing EX values for each source, difficulty (for 'bird_dev'), and overall EX.
    """
    
    # Load the test data
    test_data = load_json(gold_sql_file)
    
    # If the result path is not specified, use the latest result
    if result_path is None:
        intermediate_results_dir = Path("./results/intermediate_results")
        latest_folder = find_latest_folder(intermediate_results_dir)
        if latest_folder is None:
            raise ValueError("Cannot find the result folder.")
        result_path = intermediate_results_dir / latest_folder
    else:
        result_path = Path(result_path)
    
    # Load the generated SQL results
    generated_sql_file = result_path / "generated_sql_results.jsonl"
    generated_sql_data = load_jsonl(str(generated_sql_file))

    # Dictionary to store counts for each source and difficulty (for 'bird_dev')
    source_stats = {}

    # Global counters for overall EX calculation
    global_total_cnt = 0
    global_correct_cnt = 0

    for item in test_data:
        question_id = item.get("question_id")
        source = item.get("source", "")
        difficulty = item.get("difficulty")
        db_id = item.get("db_id", "")
        db_folder = f"{source}_{db_id}"
        db_file = f"{db_id}.sqlite"
        db_path = str(Config().database_dir / db_folder / db_file)

        gold_sql = item.get("gold_SQL")
        generated_sql = ""

        # Find the corresponding generated SQL
        for i in generated_sql_data:
            if i.get("question_id") == question_id:
                generated_sql = i.get("generated_sql")
                break

        curr_res = compare_sql_results(db_path, gold_sql, generated_sql)
        # Initialize the source in the dictionary if not already present
        if source not in source_stats:
            source_stats[source] = {"total_cnt": 0, "correct_cnt": 0}

        # For 'bird_dev', further group by difficulty
        if source == "bird_dev":
            if "difficulty_stats" not in source_stats[source]:
                source_stats[source]["difficulty_stats"] = {}
            if difficulty not in source_stats[source]["difficulty_stats"]:
                source_stats[source]["difficulty_stats"][difficulty] = {"total_cnt": 0, "correct_cnt": 0}

            source_stats[source]["difficulty_stats"][difficulty]["total_cnt"] += 1
            if curr_res:
                source_stats[source]["difficulty_stats"][difficulty]["correct_cnt"] += 1

        source_stats[source]["total_cnt"] += 1
        if curr_res:
            source_stats[source]["correct_cnt"] += 1

        # Update global counters
        global_total_cnt += 1
        if curr_res:
            global_correct_cnt += 1

    # Calculate EX for each source and difficulty (for 'bird_dev')
    ex_results = {}
    for source, stats in source_stats.items():
        total_cnt = stats["total_cnt"]
        correct_cnt = stats["correct_cnt"]
        EX = round(correct_cnt * 100 / total_cnt, 2)
        ex_results[source] = {"total_EX": EX}

        if source == "bird_dev" and "difficulty_stats" in stats:
            ex_results[source]["difficulty_EX"] = {}
            for difficulty, diff_stats in stats["difficulty_stats"].items():
                diff_total_cnt = diff_stats["total_cnt"]
                diff_correct_cnt = diff_stats["correct_cnt"]
                diff_EX = round(diff_correct_cnt * 100 / diff_total_cnt, 2)
                ex_results[source]["difficulty_EX"][difficulty] = diff_EX
                print(f"Source: {source}, Difficulty: {difficulty}, Total number of questions: {diff_total_cnt}, Number of generated SQLs executed correctly: {diff_correct_cnt}, EX value: {diff_EX}")

        print(f"Source: {source}, Total number of questions: {total_cnt}, Number of generated SQLs executed correctly: {correct_cnt}, EX value: {EX}")

    # Calculate overall EX
    overall_EX = round(global_correct_cnt * 100 / global_total_cnt, 2)
    ex_results["overall_EX"] = overall_EX
    print(f"Overall EX: Total number of questions: {global_total_cnt}, Number of generated SQLs executed correctly: {global_correct_cnt}, EX value: {overall_EX}")

    return ex_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_sql_file', type=str, required=True,
                       help='Path to file containing gold SQL queries')
    parser.add_argument('--result_path', type=str, 
                       default=None,
                       help='Path to result folder (default: latest result)')
    args = parser.parse_args()
    
    compute_EX_source_difficulty_based(args.gold_sql_file, args.result_path)

if __name__ == "__main__":
    main()