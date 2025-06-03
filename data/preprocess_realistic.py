import json

curr_id = 0

# 加载JSON文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 保存JSON文件
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# # # 处理Spider dev的对象
# def process_spider_dev(spider_dev_data):
#     global curr_id
#     processed = []
#     for item in spider_dev_data:
#         curr_id+=1
#         processed.append({
#             "question_id": curr_id,
#             "db_id": item.get("db_id", ""),
#             "question": item.get("question", ""),
#             "gold_SQL": item.get("query", ""),
#             "evidence": "NONE",
#             "difficulty": "spider",
#             "source": "spider_dev"
#         })
#     return processed

# # 处理Spider test的对象
# def process_spider_test(spider_test_data):
#     global curr_id
#     processed = []
#     for item in spider_test_data:
#         curr_id+=1
#         processed.append({
#             "question_id": curr_id,
#             "db_id": item.get("db_id", ""),
#             "question": item.get("question", ""),
#             "gold_SQL": item.get("query", ""),
#             "evidence": "NONE",
#             "difficulty": "spider",
#             "source": "spider_test"
#         })
#     return processed


# 处理BIRD的对象
# def process_bird_dev(bird_dev_data):
#     global curr_id
#     processed = []
#     for item in bird_dev_data:
#         curr_id+=1
#         processed.append({
#             "question_id": curr_id,
#             "db_id": item.get("db_id", ""),
#             "question": item.get("question", ""),
#             "gold_SQL": item.get("SQL", ""),
#             "evidence": item.get("evidence", "NONE"),
#             "difficulty": item.get("difficulty", ""),
#             "source": "bird_dev"
#         })
#     return processed

# # 处理BIRD train的对象
# def process_bird_train(bird_train_data):
#     global curr_id
#     processed = []
#     for item in bird_train_data:
#         curr_id+=1
#         processed.append({
#             "question_id": curr_id,
#             "db_id": item.get("db_id", ""),
#             "question": item.get("question", ""),
#             "gold_SQL": item.get("SQL", ""),
#             "evidence": item.get("evidence", "NONE"),
#             "difficulty": item.get("difficulty", ""),
#             "source": "bird_train"
#         })
#     return processed

# # 处理Spider realistic的对象
def process_spider_dev(spider_dev_data):
    global curr_id
    processed = []
    for item in spider_dev_data:
        curr_id+=1
        processed.append({
            "question_id": curr_id,
            "db_id": item.get("db_id", ""),
            "question": item.get("question", ""),
            "gold_SQL": item.get("query", ""),
            "evidence": "NONE",
            "difficulty": "spider",
            "source": "spider_dev"
        })
    return processed




# 合并Spider和BIRD数据
def merge_data(spider_dev_file, spider_test_file, bird_dev_file, bird_train_file, output_file):
    spider_dev_data = load_json(spider_dev_file)
    # spider_test_data = load_json(spider_test_file)
    # bird_dev_data = load_json(bird_dev_file)
    # bird_train_data = load_json(bird_train_file)

    
    processed_spider_dev = process_spider_dev(spider_dev_data)
    # processed_spider_test = process_spider_test(spider_test_data)
    # processed_bird_dev = process_bird_dev(bird_dev_data)
    # processed_bird_train = process_bird_train(bird_train_data)

    # merged_data = processed_spider_dev + processed_spider_test + processed_bird_dev

    # save_json(merged_data, output_file)
    save_json(processed_spider_dev, output_file)

# 文件路径
spider_dev_path = "data/spider/spider_realistic.json"
spider_test_path = "spider_test.json"
bird_dev_path = "bird_dev.json"
bird_train_path = "bird_train.json"

output_file_path = "formatted_spider_realistic.json"



# 合并数据
merge_data(spider_dev_path, spider_test_path, bird_dev_path, bird_train_path, output_file_path)

print(f"合并完成，结果已保存到 {output_file_path}")

# python -m data.preprocess_new