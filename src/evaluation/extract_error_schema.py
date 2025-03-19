import json

# 文件路径
file1_path = "results/raw_results/qwen_classifier_sft/error_sql_results.jsonl"       # error sql路径
file2_path = "results/raw_results/qwen_classifier_sft/linked_schema_results.jsonl"     # linked schema 路径
output_file_path = "results/raw_results/qwen_classifier_sft/error_schema_results.jsonl"      # 输出文件路径（文件3）

# 先读取文件2，将所有对象存入字典，以 query_id 为 key 方便查找
query_mapping = {}
with open(file2_path, "r", encoding="utf-8") as f2:
    for line in f2:
        if line.strip():  # 排除空行
            obj = json.loads(line)
            query_id = obj.get("query_id")
            if query_id is not None:
                query_mapping[query_id] = obj

# 遍历文件1，根据 question_id 查找文件2中对应的对象，并写入输出文件
with open(file1_path, "r", encoding="utf-8") as f1, open(output_file_path, "w", encoding="utf-8") as fout:
    for line in f1:
        if line.strip():
            obj1 = json.loads(line)
            question_id = obj1.get("question_id")
            if question_id in query_mapping:
                matched_obj = query_mapping[question_id]
                fout.write(json.dumps(matched_obj, ensure_ascii=False) + "\n")
