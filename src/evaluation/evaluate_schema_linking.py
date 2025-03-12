import json
from ..core.utils import load_json, load_jsonl
from ..core.config import Config
import sqlite3
import os
from pathlib import Path

def calculate_precision(golden_files, dev_file):
    # 加载文件内容
    golden_data_spider_dev = load_json(golden_files[0])
    golden_data_spider_test = load_json(golden_files[1])
    golden_data_bird_dev = load_json(golden_files[2])
    
    dev_data = load_jsonl(dev_file)
    
    correct_links = 0  # 预测正确的表/列数
    total_predictions = 0  # 预测的表/列总数

    # 遍历每个问题
    cnt = 0

    for dev in (dev_data):
        # 获取 Dev 的表和列
        dev_links = set(
            (table["table"].lower(), col.lower()) 
            for table in dev["tables"] 
            for col in table["columns"]
            if col is not None  # 添加空值检查
        )

        query_id = dev["query_id"]
        gold_query_id = -1

        if query_id <= 1034:
            golden_data = golden_data_spider_dev
            gold_query_id = query_id
        elif query_id <= 3181:
            golden_data = golden_data_spider_test
            gold_query_id = query_id - 1035
        else: # ( <= 4715)
            golden_data = golden_data_bird_dev
            gold_query_id = query_id - 3182

        if(gold_query_id < 0):
            print("gold_query_id < 0，出现异常")

        # 通过gold_query_id，在golden_data中遍历找到对应的对象
        for golden in golden_data:
            if golden["id"] == gold_query_id:
                # is_count_star = "count(*)" in golden["gold_sql"].lower()
                golden_links = set()
                for table in golden["tables"]:
                    table_name = table["table"].lower()
                    #if not is_count_star or table["columns"]:  # 仅当没有 COUNT(*) 或列非空时才记录
                    golden_links.update((table_name, col.lower()) for col in table["columns"])
                        # 如果是 COUNT(*)，所有 dev_links 都视为正确
                # if is_count_star:
                #     correct_links += len(dev_links)
                # else: # 统计正确预测和总预测
                correct_links += len(golden_links & dev_links)  # 交集的数量
                total_predictions += len(dev_links)  # Dev 预测的数量
                
                cnt+=1
                if(cnt %100 == 34) :
                    print(f'处理了{cnt}个, correct links {correct_links}, total prediction links {total_predictions}')
                break
    # 计算精确度
    precision = correct_links / total_predictions if total_predictions > 0 else 0
    return precision    


def calculate_recall(golden_file, dev_file):
    # 加载文件内容
    golden_data_spider_dev = load_json(golden_files[0])
    golden_data_spider_test = load_json(golden_files[1])
    golden_data_bird_dev = load_json(golden_files[2])
    
    dev_data = load_jsonl(dev_file)
    
    correct_links = 0  # 预测正确的表/列数
    total_golden_links = 0  # 实际相关的表/列总数

    # 遍历每个问题
    cnt = 0

    for dev in (dev_data):
        # 获取 Dev 的表和列
        dev_links = set(
            (table["table"].lower(), col.lower()) 
            for table in dev["tables"] 
            for col in table["columns"]
            if col is not None  # 添加空值检查
        )

        query_id = dev["query_id"]
        gold_query_id = -1

        if query_id <= 1034:
            golden_data = golden_data_spider_dev
            gold_query_id = query_id
        elif query_id <= 3181:
            golden_data = golden_data_spider_test
            gold_query_id = query_id - 1035
        else: # ( <= 4715)
            golden_data = golden_data_bird_dev
            gold_query_id = query_id - 3182

        if(gold_query_id < 0):
            print("gold_query_id < 0，出现异常")

        # 通过gold_query_id，在golden_data中遍历找到对应的对象
        for golden in golden_data:
            if golden["id"] == gold_query_id:
                # 检查是否存在 COUNT(*)
                # is_count_star = "count(*)" in golden["gold_sql"].lower()
                # 获取 Golden 的表和列，并标准化为小写
                golden_links = set()
                for table in golden["tables"]:
                    table_name = table["table"].lower()
                    #if not is_count_star or table["columns"]:  # 仅当没有 COUNT(*) 或列非空时才记录
                    golden_links.update((table_name, col.lower()) for col in table["columns"])

                # 如果是 COUNT(*)，只考虑表级别
                # if is_count_star:
                #     correct_links += len(dev_links)  # 所有 dev_links 视为正确
                #     total_golden_links += len(dev_links)  # 认为 golden 中也有同等数量的列
                # else:
                # 统计正确预测和标准答案
                correct_links += len(golden_links & dev_links)  # 交集的数量
                total_golden_links += len(golden_links)  # Golden 的总数

                cnt+=1
                if(cnt %100 == 34) :
                    print(f'处理了{cnt}个, correct links {correct_links}, total golden links {total_golden_links}')
                break
    # 计算召回率
    recall = correct_links / total_golden_links if total_golden_links > 0 else 0
    return recall


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0  # 避免除以零
    return 2 * (precision * recall) / (precision + recall)


def count_origin_columns(dev_file):
    dev_data = load_json(dev_file)

    total_columns = 0

    for item in dev_data:
        question_id = item.get("question_id")
        source = item.get("source", "")
        db_id = item.get("db_id", "")
        db_folder = f"{source}_{db_id}"
        db_file = f"{db_id}.sqlite"
        db_path = str(Config().database_dir / db_folder / db_file)
        total_columns += count_total_columns(db_path)

    return total_columns


def count_total_columns(db_path):
    """
    统计 SQLite 数据库中所有表的列总数。
    :param db_path: SQLite 数据库文件路径
    :return: 数据库中的总列数
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_columns = 0
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        columns = cursor.fetchall()
        total_columns += len(columns)
    
    conn.close()
    return total_columns


def calculate_nsr_srr(golden_files, dev_file):
    # 加载文件内容
    golden_data_spider_dev = load_json(golden_files[0])
    golden_data_spider_test = load_json(golden_files[1])
    golden_data_bird_dev = load_json(golden_files[2])
    
    dev_data = load_jsonl(dev_file)
    
    total_correct_links = 0  # 预测正确的表/列数 (用于 NSR)
    total_golden_links = 0  # 真实 schema 元素总数 (用于 NSR)
    total_strict_correct = 0  # 严格召回成功的个数 (用于 SRR)
    total_questions = len(dev_data)  # 问题总数 (用于 SRR)

    # 遍历每个问题
    for dev in dev_data:
        # 获取 Dev 预测的表和列
        dev_links = set(
            (table["table"].lower(), col.lower()) 
            for table in dev["tables"] 
            for col in table["columns"]
            if col is not None  # 过滤掉 None 值
        )

        query_id = dev["query_id"]
        gold_query_id = -1

        if query_id <= 1034:
            golden_data = golden_data_spider_dev
            gold_query_id = query_id
        elif query_id <= 3181:
            golden_data = golden_data_spider_test
            gold_query_id = query_id - 1035
        else:  # query_id <= 4715
            golden_data = golden_data_bird_dev
            gold_query_id = query_id - 3182

        if gold_query_id < 0:
            print("gold_query_id < 0，出现异常")
            continue

        # 在 golden_data 中找到对应的问题
        for golden in golden_data:
            if golden["id"] == gold_query_id:
                # 真实的 schema 元素集合
                golden_links = set(
                    (table["table"].lower(), col.lower()) 
                    for table in golden["tables"] 
                    for col in table["columns"]
                )

                # NSR 计算
                total_correct_links += len(golden_links & dev_links)  # 交集
                total_golden_links += len(golden_links)  # 标准答案总数

                # SRR 计算
                if golden_links.issubset(dev_links):  # 如果 golden_links ⊆ dev_links
                    total_strict_correct += 1

                break  # 找到对应的 golden 数据就停止

    # 计算 NSR
    nsr = total_correct_links / total_golden_links if total_golden_links > 0 else 0

    # 计算 SRR
    srr = total_strict_correct / total_questions if total_questions > 0 else 0

    return nsr, srr





if __name__ == '__main__':
    golden_file_spider_dev = str(Config().gold_schema_linking_dir / "spider_dev_gold_schema_linking.json")
    golden_file_spider_test = str(Config().gold_schema_linking_dir / "spider_test_gold_schema_linking.json")
    golden_file_bird_dev = str(Config().gold_schema_linking_dir / "bird_dev_gold_schema_linking.json")
    

    golden_files = []
    golden_files.append(golden_file_spider_dev)
    golden_files.append(golden_file_spider_test)
    golden_files.append(golden_file_bird_dev)
    
    # dev_file = str(Config().results_dir / "raw_results/knn_20250212_211131/linked_schema_results.jsonl")
    dev_file = str(Config().results_dir / "raw_results/knn_20250212_211131/error_schema_results.jsonl")

    # dev_file_count_origin = "./data/sampled_merged.json"
    dev_file_count_origin = "data/formatted_bird_dev.json"


    precision = calculate_precision(golden_files, dev_file)
    recall = calculate_recall(golden_files, dev_file)
    print(f"Schema Linking Precision: {precision:.4f}")
    print(f"Schema Linking Recall: {recall:.4f}")

    f1_score = calculate_f1(precision, recall)
    print(f"F1 Score: {f1_score:.4f}")

    origin_columns = count_origin_columns(dev_file_count_origin)
    print(f"每个问题对应的数据库原始共有 {origin_columns} 列")

    nsr, srr = calculate_nsr_srr(golden_files, dev_file)

    print(f"nsr(非严格召回率)： {nsr:.4f}")
    print(f"srr(严格召回率)： {srr:.4f}")



