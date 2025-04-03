import json

# lode JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# save JSON file
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# process Spider's objects
def process_spider(spider_data):
    processed = []
    for item in spider_data:
        processed.append({
            "db_id": item.get("db_id", ""),
            "question": item.get("question", ""),
            "SQL": item.get("query", ""),
            "evidence": "NONE",
            "difficulty": "spider",
            "source": "spider"
        })
    return processed

# process BIRD's objects
def process_bird(bird_data):
    processed = []
    for item in bird_data:
        processed.append({
            "db_id": item.get("db_id", ""),
            "question": item.get("question", ""),
            "SQL": item.get("SQL", ""),
            "evidence": item.get("evidence", "NONE"),
            "difficulty": item.get("difficulty", ""),
            "source": "bird"
        })
    return processed

# merge Spider and BIRD data
def merge_data(spider_file, bird_file, output_file):
    spider_data = load_json(spider_file)
    bird_data = load_json(bird_file)

    processed_spider = process_spider(spider_data)
    processed_bird = process_bird(bird_data)

    merged_data = processed_spider + processed_bird

    save_json(merged_data, output_file)


if __name__ == "__main__":
    """
    Process the Spider and Bird datasets into the format required for this project
    """
    
    # Example file path
    spider_file_path = "spider_dev.json"
    bird_file_path = "bird_dev.json"
    output_file_path = "merged_dev.json"

    # merge data
    merge_data(spider_file_path, bird_file_path, output_file_path)

    print(f"Merge completed, results saved to {output_file_path}")
