import json

# 定义文件路径
#file1_path = "results/intermediate_results/20250220_061646/DPOClassifierRouter.jsonl"
# file1_path = "results/raw_results/qwen_cascade/QwenCascadeRouter.jsonl"
# file1_path = "results/raw_results/qwen_classifier_sft/QwenClassifierRouter.jsonl"
# file1_path = "results/raw_results/qwen_pairwise_sft/QwenPairwiseRouter.jsonl"
# file1_path = "results/raw_results/roberta_classifier_sft/RoBERTaClassifierRouter.jsonl"
file1_path = "results/raw_results/knn_20250212_211131/KNNClassifierRouter.jsonl"
file2_path = "results/intermediate_results/20250305_111304/generated_sql_results.jsonl"
file3_path = "results/intermediate_results/20250305_122557/generated_sql_results.jsonl"
file4_path = "results/intermediate_results/20250306_074432/generated_sql_results.jsonl"
output_path = "results/raw_results/knn_20250212_211131/haiku_sql_results.jsonl"

def load_jsonl_to_dict(file_path, key_field):
    """
    读取 jsonl 文件，并以 key_field 字段构建一个字典。
    """
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                key = obj.get(key_field)
                mapping[key] = obj
    return mapping

# 加载文件2、文件3、文件4到对应的字典中，键为 question_id
file2_dict = load_jsonl_to_dict(file2_path, "question_id")
file3_dict = load_jsonl_to_dict(file3_path, "question_id")
file4_dict = load_jsonl_to_dict(file4_path, "question_id")

# 打开输出文件，并遍历文件1中的对象，根据 selected_generator 决定从哪个文件中取出对应对象
with open(output_path, 'w', encoding='utf-8') as outfile:
    with open(file1_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            if line.strip():
                obj1 = json.loads(line)
                query_id = obj1.get("query_id")
                selected_generator = obj1.get("output", {}).get("selected_generator")
                result_obj = None

                # 根据 selected_generator 的值选择对应的字典查找对象
                if selected_generator == "basic":
                    result_obj = file2_dict.get(query_id)
                    print("basic")
                elif selected_generator == "intermediate":
                    result_obj = file3_dict.get(query_id)
                    print("intermediate")
                elif selected_generator == "advanced":
                    result_obj = file4_dict.get(query_id)
                    print("advanced")

                # 如果找到对应对象则写入输出文件
                if result_obj is not None:
                    outfile.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                else:
                    # 没有找到对应对象，可选择打印日志或者忽略
                    print(f"未找到 query_id={query_id} 对应的结果对象（selected_generator={selected_generator}）")
