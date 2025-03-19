import os
from pathlib import Path

from ..core.utils import load_json, load_jsonl
from ..core.sql_execute import *
from ..core.config import Config

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
    
    if result1.result is None:
        print("gold SQL 执行成功但结果为空。")
        if result2.result is None:
            # 如果两个SQL都返回空结果，认为它们是相等的
            return 1
        return 0

    if result2.result is None:
        print("generated SQL 执行成功但结果为空。")
        return 0
    
    # 比较结果
    if set(result1.result) == set(result2.result):
        return 1
    else:
        return 0



def compute_EX_source_difficulty_based(merge_dev_demo_file: str, result_path: str):
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





def compute_EX_pipeline_type_based(merge_dev_demo_file: str, pipeline_label_file: str, result_path: str):
    """
    根据生成的 SQL 结果，按 pipeline_type 分组计算 EX 值。
    pipeline_type 信息从专门的 pipeline_label 文件（如 bird_dev_pipeline_label.jsonl）中加载，
    merge_dev_demo_file（如 formatted_bird_dev.json）中包含 gold SQL 和问题元信息，
    result_path（如 generated_sql_results.jsonl）中包含生成的 SQL 结果。
    
    Args:
        merge_dev_demo_file: 格式化后的 NL2SQL 文件路径（如 formatted_bird_dev.json）。
        pipeline_label_file: 包含 pipeline_type 信息的文件路径（如 bird_dev_pipeline_label.jsonl）。
        result_path: 生成 SQL 结果文件路径（如 generated_sql_results.jsonl）。
        
    Returns:
        一个字典，包含各 pipeline_type 的 EX 值以及总体 EX 值。
    """
    # 加载数据
    merge_dev_demo_data = load_json(merge_dev_demo_file)
    pipeline_data = load_jsonl(pipeline_label_file)
    generated_sql_data = load_jsonl(result_path)
    
    # 构建 question_id -> pipeline_type 的映射（假设每个 question_id 在 pipeline 文件中唯一）
    pipeline_mapping = {}
    for item in pipeline_data:
        qid = item.get("question_id")
        pipeline_type = item.get("pipeline_type")
        if qid is not None and pipeline_type is not None:
            pipeline_mapping[qid] = pipeline_type
    
    # 用于存储每个 pipeline_type 的统计数据
    pipeline_stats = {}
    global_total_cnt = 0
    global_correct_cnt = 0

    for item in merge_dev_demo_data:
        question_id = item.get("question_id")
        # 根据 question_id 从 pipeline_mapping 中获取 pipeline_type，如未找到则归为 "unknown"
        pipeline_type = pipeline_mapping.get(question_id, "unknown")
        
        source = item.get("source", "")
        db_id = item.get("db_id", "")
        # 构造数据库文件路径（假设目录结构为：{source}_{db_id}/{db_id}.sqlite）
        db_folder = f"{source}_{db_id}"
        db_file = f"{db_id}.sqlite"
        db_path = str(Config().database_dir / db_folder / db_file)
        
        # 获取 gold SQL（注意可能存在 "gold_SQL" 或 "gold_sql" 字段）
        gold_sql = item.get("gold_SQL") or item.get("gold_sql")
        
        # 在 generated_sql_data 中查找对应 question_id 的生成 SQL
        generated_sql = ""
        for result_item in generated_sql_data:
            if result_item.get("question_id") == question_id:
                generated_sql = result_item.get("generated_sql")
                break
        
        # 执行 SQL 比较，返回 1（相同） 或 0（不同）
        curr_res = compare_sql_results(db_path, gold_sql, generated_sql)
        
        # 初始化当前 pipeline_type 的统计数据
        if pipeline_type not in pipeline_stats:
            pipeline_stats[pipeline_type] = {"total_cnt": 0, "correct_cnt": 0}
        
        pipeline_stats[pipeline_type]["total_cnt"] += 1
        if curr_res:
            pipeline_stats[pipeline_type]["correct_cnt"] += 1
        
        global_total_cnt += 1
        if curr_res:
            global_correct_cnt += 1
    
    # 计算每个 pipeline_type 的 EX 值
    ex_results = {}
    for p_type, stats in pipeline_stats.items():
        total = stats["total_cnt"]
        correct = stats["correct_cnt"]
        ex_value = round(correct * 100 / total, 2) if total > 0 else 0.0
        ex_results[p_type] = ex_value
        print(f"Pipeline type: {p_type}, 总计问题数量：{total}, 生成 SQL 正确执行数量：{correct}, EX 值：{ex_value}")
    
    # 计算总体 EX
    overall_EX = round(global_correct_cnt * 100 / global_total_cnt, 2) if global_total_cnt > 0 else 0.0
    ex_results["overall_EX"] = overall_EX
    print(f"Overall EX: 总计问题数量：{global_total_cnt}, 生成 SQL 正确执行数量：{global_correct_cnt}, EX 值：{overall_EX}")
    
    return ex_results







if __name__ == "__main__":
    merge_dev_demo_file = "./data/formatted_bird_dev.json"
    pipeline_label_file = "./data/labeled/bird_dev_pipeline_label.jsonl"
    result_path = "results/raw_results/dpo_20250220_061646/generated_sql_results.jsonl"

    print("结果：按 pipeline_type 统计的 EX 值")
    compute_EX_pipeline_type_based(merge_dev_demo_file, pipeline_label_file, result_path)

    #compute_EX_source_difficulty_based(merge_dev_demo_file, result_path)


    exit()