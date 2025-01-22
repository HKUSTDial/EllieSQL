import os

from ..core.utils import load_json, load_jsonl
from ..core.sql_execute import *


def find_latest_folder(directory):
    """
    Find the latest folder in intermediate results folder.
    
    Args:
        directory: The path to the intermediate_results
    Returns:
        The latest_folder.
    """ 
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    if not folders:
        return None  # 如果没有文件夹，返回 None
    
    # 按字典序排序，最后一个是最新的
    latest_folder = sorted(folders)[-1]
    return latest_folder




# def compare_sql_results_new(db_path, sql1, sql2, timeout=10):
#     """
#     Compare the ex results for two sqls.
    
#     Args:
#         db_path: The path to the .sqlite file.
#         sql1: The gold sql.
#         sql2: The generated sql.
#     Returns:
#         1 for equal, 0 for not equal.
#     """
#     # 执行第一个 SQL 语句
#     flag1, message1 = validate_sql(db_path, sql1)
#     result1 = execute_sql_with_timeout(db_path, sql1, timeout)
#     if(flag1 == False):
#         print(f"gold SQL 运行出现报错!! 错误信息如下：{message1}")
#         print(db_path)
#         print(sql1)
#         return 0
#     # 执行第二个 SQL 语句
#     flag, message = validate_sql(db_path, sql2)
#     if(flag == False):
#         print(f"generated SQL 运行出现报错。错误信息如下：{message}")
#         return 0
#     result2 = execute_sql_with_timeout(db_path, sql2, timeout)

#     # 比较结果
#     if set(result1.result) == set(result2.result):
#         return 1
#     else:
#         return 0


def compare_sql_results(db_path, sql1, sql2, timeout=10):
    """
    Compare the ex results for two sqls.
    
    Args:
        db_path: The path to the .sqlite file.
        sql1: The gold sql.
        sql2: The generated sql.
    Returns:
        1 for equal, 0 for not equal.
    """
    # 执行第一个 SQL 语句
    result1 = execute_sql_with_timeout(db_path, sql1, timeout)
    if result1 is None:
        print("gold SQL 执行失败!!!! 请检查错误信息。")
        return 0
    # 执行第二个 SQL 语句
    result2 = execute_sql_with_timeout(db_path, sql2, timeout)
    if result2 is None:
        print("generated SQL 没有产生结果对象。")
        return 0
    
    # flag, message = validate_sql(db_path, sql2)
    # if(flag == False):
    #     print(f"generated SQL 运行flag为False。False信息如下：{message}")
    #     print(db_path)
    #     print(sql2)

    if result2.result_type == SQLExecutionResultType.TIMEOUT:
        print(f"generated SQL 运行超时。错误信息如下：{result2.error_message}")
        return 0
    if result2.result_type == SQLExecutionResultType.ERROR:
        print(f"generated SQL 运行出现报错。错误信息如下：{result2.error_message}")
        return 0
    
    if result2.result is None and result2.result_type == SQLExecutionResultType.SUCCESS:
        print(f"generated SQL 运行成功但结果为空。")
    #     return 0
    #else:
        #print(f"gold SQL结果: {result1} \n generated SQL结果: {result2}")
    
    # 比较结果
    if set(result1.result) == set(result2.result):
        return 1
    else:
        return 0



def compute_EX_source_difficulty_based(merge_dev_demo_file, time_path):
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
    time_path = time_path + find_latest_folder(time_path) 

    generated_sql_file = time_path + "/" + "generated_sql_results.jsonl"
    generated_sql_data = load_jsonl(generated_sql_file)

    # Dictionary to store counts for each source and difficulty (for 'bird_dev')
    source_stats = {}

    # Global counters for overall EX calculation
    global_total_cnt = 0
    global_correct_cnt = 0

    for item in merge_dev_demo_data:
        question_id = item.get("question_id")
        source = item.get("source", "")
        difficulty = item.get("difficulty")
        db_id = item.get("db_id", "")
        db_path = "./data/merged_databases/" + source +'_'+ db_id +"/"+ db_id + '.sqlite'

        gold_sql = item.get("gold_SQL")
        generated_sql = ""

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
                print(f"Source: {source}, Difficulty: {difficulty}, 总计问题数量：{diff_total_cnt}, 生成sql正确执行数量{diff_correct_cnt}, EX值{diff_EX}")

        print(f"Source: {source}, 总计问题数量：{total_cnt}, 生成sql正确执行数量{correct_cnt}, EX值{EX}")

    # Calculate overall EX
    overall_EX = round(global_correct_cnt * 100 / global_total_cnt, 2)
    ex_results["overall_EX"] = overall_EX
    print(f"Overall EX: 总计问题数量：{global_total_cnt}, 生成sql正确执行数量{global_correct_cnt}, EX值{overall_EX}")

    return ex_results

if __name__ == "__main__":
    # merge_dev_demo_file = "./data/sampled_bird_dev.json"
    merge_dev_demo_file = "./data/sampled_merged.json"
    #merge_dev_demo_file = "./data/merge_dev_demo.json"
    time_path = "./results/intermediate_results/"

    compute_EX_source_difficulty_based(merge_dev_demo_file, time_path)
    exit()