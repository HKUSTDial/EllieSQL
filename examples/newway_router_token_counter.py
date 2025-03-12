import json

def load_jsonl(file_path):
    """
    读取 jsonl 文件，将每行 JSON 对象解析后，按 query_id 组织成字典。
    """
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                query_id = obj["query_id"]
                data[query_id] = obj
    return data

# 加载 generator 文件，分别对应 selected_generator 的不同值
generator_basic = load_jsonl("results/intermediate_results/20250305_111304/GPTSQLGenerator.jsonl")
generator_intermediate = load_jsonl("results/intermediate_results/20250305_122557/DCRefinerSQLGenerator.jsonl")
generator_advanced = load_jsonl("results/intermediate_results/20250306_074432/EnhancedSQLGenerator.jsonl")

total_input_tokens = 0
total_output_tokens = 0

# 遍历 router_result.jsonl 文件中的对象
# with open("results/intermediate_results/20250220_061646/DPOClassifierRouter.jsonl", "r", encoding="utf-8") as f:
# with open("results/raw_results/qwen_cascade/QwenCascadeRouter.jsonl", "r", encoding="utf-8") as f:
# with open("results/raw_results/qwen_classifier_sft/QwenClassifierRouter.jsonl", "r", encoding="utf-8") as f:
# with open("results/raw_results/qwen_pairwise_sft/QwenPairwiseRouter.jsonl", "r", encoding="utf-8") as f:
# with open("results/raw_results/roberta_classifier_sft/RoBERTaClassifierRouter.jsonl", "r", encoding="utf-8") as f:
with open("results/raw_results/knn_20250212_211131/KNNClassifierRouter.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            router_obj = json.loads(line)
            query_id = router_obj["query_id"]
            selected_generator = router_obj["output"]["selected_generator"]
            # 根据 selected_generator 的值选择对应的 generator 文件
            if selected_generator == "basic":
                target_obj = generator_basic.get(query_id)
            elif selected_generator == "intermediate":
                target_obj = generator_intermediate.get(query_id)
            elif selected_generator == "advanced":
                target_obj = generator_advanced.get(query_id)
            else:
                target_obj = None

            # 若找到对应的对象，则累加 input_tokens 和 output_tokens
            if target_obj:
                total_input_tokens += target_obj["model_info"]["input_tokens"]
                total_output_tokens += target_obj["model_info"]["output_tokens"]

print("total_input_tokens:", total_input_tokens)
print("total_output_tokens:", total_output_tokens)
