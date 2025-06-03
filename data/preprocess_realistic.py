import json

curr_id = 0

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Deal with Spider realistic objects
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


# Merge Spider and BIRD data
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


if __name__ == "__main__":
    # File paths
    spider_dev_path = "data/spider/spider_realistic.json"
    spider_test_path = "spider_test.json"
    bird_dev_path = "bird_dev.json"
    bird_train_path = "bird_train.json"

    output_file_path = "formatted_spider_realistic.json"

    # Merge data
    merge_data(spider_dev_path, spider_test_path, bird_dev_path, bird_train_path, output_file_path)

    print(f"Merge completed, results saved to {output_file_path}")

    # python -m data.preprocess_realistic