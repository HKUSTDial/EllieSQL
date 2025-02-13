import re
import json

# 读取文件并提取 id
def extract_ids(file_path):
    ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'id (\d+):', line)
            if match:
                ids.append(int(match.group(1)))  # 提取数字并转换为整数
    return ids


# 读取 input.json 并筛选数据
def filter_json(input_file, output_file, id_list):
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    origin_num = len(data)
    # 过滤不在 id_list 里的对象
    filtered_data = [item for item in data if item["question_id"] not in id_list]

    new_num = len(filtered_data)
    deleted_num = origin_num - new_num

    # 将筛选后的数据写入 output.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"过滤完成，总共{origin_num}个对象，删除了{deleted_num}个，留下了{new_num}个")



if __name__ == "__main__":
    # 示例调用
    empty_file_path = "./data/bird_train_gold_empty_count.txt"  # 请替换为你的文件路径
    id_list = extract_ids(empty_file_path)

    print(id_list)

    origin_dataset_file = "./data/formatted_bird_train.json"
    output_dataset_file = "./data/nonempty_formatted_bird_train.json"

    filter_json(origin_dataset_file, output_dataset_file, id_list)
    
