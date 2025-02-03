import os

from ..core.utils import load_json, load_jsonl
from ..core.sql_execute import *
from .compute_EX import compare_sql_results
from ..core.config import Config


# 统计结果
def analyze_results(direct_selector_path, sampled_merged_path):
    # 读取 sampled_merged.json 文件
    sampled_merged = load_json(sampled_merged_path)
    selector_data = load_jsonl(direct_selector_path)

    # 初始化统计变量
    total_objects = 0
    both_correct = 0
    one_correct = 0
    both_wrong = 0
    only_one_candidate_and_correct = 0
    only_one_candidate_and_wrong = 0

    # 读取 DirectSelector.jsonl 文件
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
                # 比较两个 candidate_sqls 的执行结果与 gold_SQL
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
            

    # 计算占比
    both_correct_ratio = both_correct / total_objects
    one_correct_ratio = one_correct / total_objects
    both_wrong_ratio = both_wrong / total_objects
    only_one_candidate_and_correct_ratio = only_one_candidate_and_correct / total_objects
    only_one_candidate_and_wrong_ratio = only_one_candidate_and_wrong / total_objects
    
    # 输出结果
    print(f"对象总数: {total_objects}")
    print(f"两个SQL执行结果都与gold_SQL相同的对象数量: {both_correct} (占比: {both_correct_ratio:.2%})")
    print(f"只有一个SQL执行结果与gold_SQL相同的对象数量: {one_correct} (占比: {one_correct_ratio:.2%})")
    print(f"两个SQL执行结果都错的对象数量: {both_wrong} (占比: {both_wrong_ratio:.2%})")
    print(f"只有一个候选sql的、并且结果正确的对象数量{only_one_candidate_and_correct}(占比：{only_one_candidate_and_correct_ratio:.2%})")
    print(f"只有一个候选sql的、并且结果错误的对象数量{only_one_candidate_and_wrong}(占比：{only_one_candidate_and_wrong_ratio:.2%})")

if __name__ == "__main__":
    direct_selector_path = "./results\important_results\\20250114_003759\DirectSelector.jsonl"
    #merge_dev_demo_file = "./data/merge_dev_demo.json"
    sampled_merged_path = "./data/sampled_merged.json"

    analyze_results(direct_selector_path, sampled_merged_path)
    exit()
# 调用函数进行分析
