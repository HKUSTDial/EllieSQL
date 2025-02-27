import json

# 读取原始数据集
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 生成偏好对
def generate_preference_pairs(data):
    preference_pairs = []
    for item in data:
        input_text = item["text"]
        label = item["label"]
        question_id = item["question_id"]
        source = item["source"]


        
        if label == 0:
            pairs = [("0", "1"), ("0", "2"), ("1", "2")]
        elif label == 1:
            pairs = [("1", "2"), ("1", "0"), ("2", "0")]
        elif label == 2:
            pairs = [("2", "0"), ("2", "1")]
        else:
            continue


        
        for chosen, rejected in pairs:
            preference_pairs.append({
                "text": input_text,
                "label": label,
                "question_id": question_id,
                "source": source,
                "chosen": chosen,
                "rejected": rejected
            })
    
    return preference_pairs

# 保存为JSON文件
def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 主函数
def main():
    input_file = "data/sft/bird_train_dataset_simplified/classifier_train.json"  # 你的原始数据文件
    output_file = "data/dpo/bird_train_dataset_simplified/classifier_train.json"  # 生成的训练数据文件
    
    data = load_json(input_file)
    preference_pairs = generate_preference_pairs(data)
    save_json(preference_pairs, output_file)
    
    print(f"已生成 {len(preference_pairs)} 个偏好对，保存在 {output_file}")

if __name__ == "__main__":
    main()
