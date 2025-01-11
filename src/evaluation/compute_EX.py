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
    
    if result2.result is None:
        if result2.result_type == SQLExecutionResultType.TIMEOUT:
            print(f"generated SQL 运行超时。错误信息如下：{result2.error_message}")
        elif result2.result_type == SQLExecutionResultType.ERROR:
            print(f"generated SQL 运行出现报错。错误信息如下：{result2.error_message}")
        elif result2.result_type == SQLExecutionResultType.ERROR:
            print(f"generated SQL 运行成功但结果为空。")
        return 0
    #else:
        #print(f"gold SQL结果: {result1} \n generated SQL结果: {result2}")
    
    # 比较结果
    if set(result1.result) == set(result2.result):
        return 1
    else:
        return 0


def compute_EX(merge_dev_demo_file, time_path):
    """
    Compute the EX for the latest generated sql results.
    
    Args:
        merge_dev_demo_file: The path to the nl2sql file.
        time_path: The path to intermediate_results (upper level catalog for such as 20250103_172805).
    Returns:
        EX.
    """
    
    merge_dev_demo_data = load_json(merge_dev_demo_file)
    time_path = time_path + find_latest_folder(time_path) 

    generated_sql_file = time_path + "/" + "generated_sql_results.jsonl"
    generated_sql_data = load_jsonl(generated_sql_file)


    total_cnt = 0
    correct_cnt = 0
    for item in merge_dev_demo_data:
        total_cnt+=1

        question_id = item.get("question_id")
        source = item.get("source", "")
        db_id = item.get("db_id", "")
        db_path = "./data/merged_databases/" + source +'_'+ db_id +"/"+ db_id + '.sqlite'

        print(db_path)

        gold_sql = item.get("gold_SQL")
        generated_sql = ""

        for i in generated_sql_data:
            if i.get("question_id") == question_id:
                generated_sql = i.get("generated_sql")
        correct_cnt += compare_sql_results(db_path, gold_sql, generated_sql)
        

    EX = round(correct_cnt*100/total_cnt, 2)
    print(f"总计问题数量：{total_cnt}, 生成sql正确执行数量{correct_cnt}, EX值{EX}")
    return


if __name__ == "__main__":
    merge_dev_demo_file = "./data/sampled_merged.json"
    #merge_dev_demo_file = "./data/merge_dev_demo.json"
    time_path = "./results/intermediate_results/"

    compute_EX(merge_dev_demo_file, time_path)
    exit()