import json
import argparse
from ..core.utils import load_jsonl

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled_file', type=str, required=True,
                       help='Path to the dataset with labells which is to be used for SFT dataset preparation')
    parser.add_argument('--preference_file', type=str, required=True,
                       help='Path to the dataset with preference pairs which is to be used for DPO dataset preparation')
    args = parser.parse_args()

    labeled_file = args.labeled_file
    preference_file = args.preference_file


    data = load_json(labeled_file)
    preference_pairs = generate_preference_pairs(data)
    save_json(preference_pairs, preference_file)
    
    print(f"已生成 {len(preference_pairs)} 个偏好对，保存在 {preference_file}")

if __name__ == "__main__":
    main()
