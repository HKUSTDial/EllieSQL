
from ..core.utils import load_json, load_jsonl, TextExtractor

from .compute_ex import execute_sql_with_timeout
from ..core.sql_execute import *

import asyncio
import json
from tqdm import tqdm
from ..core.llm import LLMBase
from ..core.config import Config
from .error_analysis_prompt import ERROR_ANALYSIS_SYSTEM, ERROR_ANALYSIS_USER
import argparse



# 1. 加载错误的结果文件
def update_error_value(file_path):
    """
    遍历指定的 JSON Lines 文件，对每个对象新增一个键 "error_value"，默认值为1，
    并将更新后的对象写回到原文件中。

    参数：
        file_path (str): JSON Lines 文件路径。
    """
    # 读取文件中的所有行
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 遍历每一行，更新 JSON 对象
    updated_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        obj = json.loads(line)
        obj["error_value"] = 1  # 新增 "error_value" 键，默认值为1
        updated_lines.append(json.dumps(obj, ensure_ascii=False))
    
    # 将更新后的对象写回原文件
    with open(file_path, "w", encoding="utf-8") as f:
        for line in updated_lines:
            f.write(line + "\n")


# 2.1 判断有没有schema linking错误
def check_sl_error(golden_file_bird_dev, error_sl_file, error_results_file):

    # 读取 schema 文件中的所有 JSON 对象
    with open(error_sl_file, 'r', encoding='utf-8') as f:
        schema_results = [json.loads(line) for line in f if line.strip()]
    
    # 读取 sql 文件中的所有 JSON 对象
    with open(error_results_file, 'r', encoding='utf-8') as f:
        sql_results = [json.loads(line) for line in f if line.strip()]

    # 构造一个字典，方便根据 question_id 快速查找 sql 对象
    sql_dict = {obj["question_id"]: obj for obj in sql_results}

    error_cnt = 0

    # 遍历 schema 文件中的每个对象
    for schema_obj in schema_results:
        # 调用已有处理函数
        result = check_sl_function(golden_file_bird_dev, schema_obj)
        # 如果返回值为 1，则进行更新
        if result == 1:
            query_id = schema_obj.get("query_id")
            if query_id in sql_dict:
                # 更新 error_value（乘以 2）
                if sql_dict[query_id]["error_value"] == 1:
                    error_cnt += 1
                    sql_dict[query_id]["error_value"] *= 2
    
    # 更新 sql_results 列表（如果需要保持原有顺序）
    updated_sql_results = []
    for obj in sql_results:
        # 根据 question_id 从字典中获取更新后的对象
        updated_sql_results.append(sql_dict[obj["question_id"]])
    
    # 将更新后的 sql 对象写回文件，每个对象占一行 JSON
    with open(error_results_file, 'w', encoding='utf-8') as f:
        for obj in updated_sql_results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"schema linking错误：{error_cnt}个")


def check_sl_function(golden_file_bird_dev, schema_obj):
    golden_data_bird_dev = load_json(golden_file_bird_dev)
    dev_links = set(
        (table["table"].lower(), col.lower()) 
        for table in schema_obj["tables"] 
        for col in table["columns"]
        if col is not None  # 过滤掉 None 值
    )

    query_id = schema_obj["query_id"]
    gold_query_id = -1

    golden_data = golden_data_bird_dev
    gold_query_id = query_id - 3182

    if gold_query_id < 0:
        print("gold_query_id < 0，出现异常")
        return 1
    
    # 在 golden_data 中找到对应的问题
    for golden in golden_data:
        if golden["id"] == gold_query_id:
            # 真实的 schema 元素集合
            golden_links = set(
                (table["table"].lower(), col.lower()) 
                for table in golden["tables"] 
                for col in table["columns"]
            )
            # SRR 计算
            if golden_links.issubset(dev_links):  # 如果 golden_links ⊆ dev_links
                return 0
            
            return 1
            break  # 找到对应的 golden 数据就停止


# 2.2 判断有没有语法错误
def check_valid_error(merge_dev_demo_file, error_results_path):
    """
    遍历文件中的每个 JSON 对象。如果对象的 "error_value" 为 1，
    则调用已有的处理函数。如果处理函数返回值为 3，则将 "error_value" 乘以 3，
    最后将所有对象保存回原文件（假设每行一个 JSON 对象）。
    """
    updated_objects = []
    valid_cnt = 0

    # 读取文件中的每一行，并解析成 JSON 对象
    with open(error_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"解析 JSON 出错: {e}")
                continue

            # 如果 error_value 为 1，则调用已有处理函数
            if obj.get("error_value") == 1:
                qid =obj["question_id"]
                db_path = ""
                with open(merge_dev_demo_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if item.get("question_id") == qid:
                            source = item.get("source", "")
                            db_id = item.get("db_id", "")
                            db_folder = f"{source}_{db_id}"
                            db_file = f"{db_id}.sqlite"
                            db_path = str(Config().database_dir / db_folder / db_file)
                            break
                result = check_sql_run(db_path, obj["generated_sql"])
                if result == 3:
                    valid_cnt += 1
                    # 将 error_value 乘以 3
                    obj["error_value"] = obj["error_value"] * 3
            updated_objects.append(obj)

    # 将更新后的对象写回到原文件，每行一个 JSON 对象
    with open(error_results_path, 'w', encoding='utf-8') as f:
        for obj in updated_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"语法错误：{valid_cnt}个")


def check_sql_run(db_path, sql2, timeout=10):
    """
    Check whether sql can run.
    Args:
        db_path: The path to the .sqlite file.
        sql2: The generated sql.
    Returns:
        1 for can run, 3 for not runnable.
    """
    # 执行第二个 SQL 语句
    result2 = execute_sql_with_timeout(db_path, sql2, timeout)
    if result2 is None:
        # print("generated SQL 没有产生结果对象。")
        return 3

    if result2.result_type == SQLExecutionResultType.TIMEOUT:
        # print(f"generated SQL 运行超时。错误信息如下：{result2.error_message}")
        return 3
    if result2.result_type == SQLExecutionResultType.ERROR:
        # print(f"generated SQL 运行出现报错。错误信息如下：{result2.error_message}")
        return 3

    if result2.result is None:
        # print("generated SQL 执行成功但结果为空。")
        return 3
    
    # 比较结果
    return 1


async def check_other_error(golden_file_bird_dev, merge_dev_demo_file, error_results_path, model, temperature, max_tokens, name):
    """
    并发处理错误分析，每次最多允许 10 个并发任务运行
    """
    llm = LLMBase()
    extractor = TextExtractor()

    # 计数列表，索引对应各类别
    my_list = [-1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, 0]

    # 读取开发集和 schema 数据
    with open(merge_dev_demo_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    with open(golden_file_bird_dev, 'r', encoding='utf-8') as f:
        golden_data_bird_dev = json.load(f)

    # 读取 error_results 文件中的所有非空行
    with open(error_results_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    updated_objects = []

    # 定义并发控制信号量，限制并发数为 10
    semaphore = asyncio.Semaphore(20)

    async def process_obj(obj):
        async with semaphore:
            # 仅对 error_value 为 1 的记录进行处理
            if obj.get("error_value") == 1:
                oid = obj["question_id"]
                question = ""
                gold_sql = ""
                # 在开发集数据中查找对应记录
                for item in dev_data:
                    if item.get("question_id") == oid:
                        question = item.get("question", "")
                        gold_sql = item.get("gold_SQL")
                        break

                # 在 golden 数据中查找对应 schema（注意：这里假设 id 与 question_id 之间存在一定偏移）
                gold_schema = None
                for i in golden_data_bird_dev:
                    if i.get("id") == oid - 3182:
                        gold_schema = i.get("tables")
                        break

                # 构造调用 llm 的 prompt
                check_error_prompt = [
                    {"role": "system", "content": ERROR_ANALYSIS_SYSTEM},
                    {"role": "user", "content": ERROR_ANALYSIS_USER.format(
                        question=question,
                        generated_sql=obj["generated_sql"],
                        golden_sql=gold_sql,
                        golden_schema=gold_schema
                    )}
                ]

                # 异步调用 llm 接口
                check_error_result = await llm.call_llm(
                    check_error_prompt,
                    model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    module_name=name
                )

                # 提取错误分类
                category_list = extractor.extract_sub_questions(check_error_result['response'])
                try:
                    category = int(category_list[0])
                except (IndexError, ValueError):
                    print("提取错误分类失败")
                    print(check_error_result)
                    return obj

                if category not in (5, 7, 11, 13):
                    print(f"错误分类错误 {category}")
                else:
                    my_list[category] += 1
                    obj["error_value"] *= category
            return obj

    # 为每个 JSON 对象创建一个任务
    tasks = [asyncio.create_task(process_obj(json.loads(line))) for line in lines]

    # 使用 tqdm 显示进度，并等待所有任务完成
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理分析进度"):
        updated_obj = await future
        updated_objects.append(updated_obj)

    # 将所有更新后的对象写回文件
    with open(error_results_path, 'w', encoding='utf-8') as f:
        for obj in updated_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print("错误分析分类完成")
    print(f"JOIN 相关错误 {my_list[5]} 个")
    print(f"GROUP BY 相关错误 {my_list[7]} 个")
    print(f"嵌套查询和集合操作错误 {my_list[11]} 个")
    print(f"其他错误 {my_list[13]} 个")



if __name__ == "__main__":
    """
    Evaluate steps:
    1. extract error sql results (extract_error_results.py)
    2. extract corresponding linked schema (extract_error_schema.py)
    3. do error analysis (error_analysis.py)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_sql_path', type=str, required=True,
                       help='the error_sql_results.jsonl file')
    parser.add_argument('--merge_dev_demo_file', type=str, required=True,
                       help='the nl2sql data file')
    parser.add_argument('--error_schema_results', type=str, required=True,
                       help='the output path for generated error_schema_results.jsonl')
    args = parser.parse_args()

    error_sql_path=args.error_sql_path
    error_schema_results=args.error_schema_results
    merge_dev_demo_file=args.merge_dev_demo_file

    update_error_value(error_sql_path)

# 2.2 判断有没有语法错误
    check_valid_error(merge_dev_demo_file, error_sql_path)

# 2.1 判断有没有schema linking错误
    # error_schema_results = "results/raw_results/qwen_classifier_sft/error_schema_results.jsonl"
    golden_file_bird_dev = str(Config().gold_schema_linking_dir / "bird_dev_gold_schema_linking.json")
    check_sl_error(golden_file_bird_dev, error_schema_results, error_sql_path)


# 2.3 判断有没有其他错误
    model = "gpt-4o-mini-2024-07-18"
    temperature = 0.0
    max_tokens = 10000
    name = "error_checker"
    golden_file_bird_dev = str(Config().gold_schema_linking_dir / "bird_dev_gold_schema_linking.json")
    asyncio.run(check_other_error(golden_file_bird_dev, merge_dev_demo_file, error_sql_path, model, temperature, max_tokens, name))

    
