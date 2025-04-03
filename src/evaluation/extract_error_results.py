import os
from pathlib import Path

from ..core.utils import load_json, load_jsonl
from ..core.sql_execute import *
from ..core.config import Config
from .compute_ex import compare_sql_results
import argparse
import json


def extract_error_results(merge_dev_demo_file: str, result_path: str, output_path: str):
    """
    Compute the EX for the latest generated sql results, grouped by source.
    For 'bird_dev' source, further group by difficulty.
    Also compute the overall EX across all sources and difficulties.
    
    Args:
        merge_dev_demo_file: The path to the nl2sql file.
        time_path: The path to intermediate_results (upper level catalog for such as 20250103_172805).
    Returns:
        A dictionary containing EX values for each source, difficulty (for 'bird_dev'), and overall EX.
    """
    
    merge_dev_demo_data = load_json(merge_dev_demo_file)

    # 使用配置路径
    generated_sql_file = Path(result_path) # / "generated_sql_results.jsonl"
    generated_sql_data = load_jsonl(str(generated_sql_file))

    # Dictionary to store counts for each source and difficulty (for 'bird_dev')
    source_stats = {}

    # Global counters for overall EX calculation
    global_total_cnt = 0
    global_correct_cnt = 0

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in merge_dev_demo_data:
            question_id = item.get("question_id")
            source = item.get("source", "")
            difficulty = item.get("difficulty")
            db_id = item.get("db_id", "")
            db_folder = f"{source}_{db_id}"
            db_file = f"{db_id}.sqlite"
            db_path = str(Config().database_dir / db_folder / db_file)

            gold_sql = item.get("gold_SQL")
            generated_sql = ""
            it=None


            for i in generated_sql_data:
                if i.get("question_id") == question_id:
                    generated_sql = i.get("generated_sql")
                    it = i
                    break


            curr_res = compare_sql_results(db_path, gold_sql, generated_sql)
            if(not curr_res):
                outfile.write(json.dumps(it, ensure_ascii=False) + "\n")
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
                print(f"Source: {source}, Difficulty: {difficulty}, 总计问题数量：{diff_total_cnt}, 生成sql正确执行数量{diff_correct_cnt}, EX值{diff_EX}")

        print(f"Source: {source}, 总计问题数量：{total_cnt}, 生成sql正确执行数量{correct_cnt}, EX值{EX}")

    # Calculate overall EX
    overall_EX = round(global_correct_cnt * 100 / global_total_cnt, 2)
    ex_results["overall_EX"] = overall_EX
    print(f"Overall EX: 总计问题数量：{global_total_cnt}, 生成sql正确执行数量{global_correct_cnt}, EX值{overall_EX}")

    return ex_results


def main():
    """
    Extract the objects whose generation results are wrong from the result file.  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True,
                       help='the generated_sql_results.jsonl file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='the extracted new file, error_sql_results.jsonl file')
    parser.add_argument('--merge_dev_demo_file', type=str, required=True,
                       help='the nl2sql data file ')
    args = parser.parse_args()

    output_path = args.output_path
    merge_dev_demo_file = args.merge_dev_demo_file
    result_path = args.result_path
    
    extract_error_results(merge_dev_demo_file, result_path, output_path)


if __name__ == "__main__":
    main()

    exit()