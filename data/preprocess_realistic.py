import json


"""
spider_realistic is a derivative dataset of spider, with each object sharing the same format as the spider_dev dataset and also being based on the spider_dev database.
Therefore, the same code used to process spider_dev can be applied to handle spider_realistic.
"""

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
            "source": "spider_dev" # spider_realistic is based on the spider_dev databases, so source is "spider_dev"
        })
    return processed


# convert Spider realistic to formatted_spider_realistic.json
def merge_data(spider_dev_file, output_file):
    spider_dev_data = load_json(spider_dev_file)

    processed_spider_dev = process_spider_dev(spider_dev_data)

    save_json(processed_spider_dev, output_file)


if __name__ == "__main__":
    # File paths
    spider_dev_path = "data/spider/spider_realistic.json"
    output_file_path = "formatted_spider_realistic.json"

    # convert Spider realistic to formatted_spider_realistic.json
    merge_data(spider_dev_path, output_file_path)

    print(f"Merge completed, results saved to {output_file_path}")

    # python -m data.preprocess_realistic