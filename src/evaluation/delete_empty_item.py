import re
import json

# Read the file and extract ids
def extract_ids(file_path):
    ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'id (\d+):', line)
            if match:
                ids.append(int(match.group(1)))  # Extract the number and convert to an integer
    return ids


# Read the input.json and filter data
def filter_json(input_file, output_file, id_list):
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    origin_num = len(data)
    # Filter objects not in id_list
    filtered_data = [item for item in data if item["question_id"] not in id_list]

    new_num = len(filtered_data)
    deleted_num = origin_num - new_num

    # Write the filtered data to output.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Filtering completed, {origin_num} objects, deleted {deleted_num}, remaining {new_num}")



if __name__ == "__main__":
    # Example call
    empty_file_path = "./data/bird_train_gold_empty_count.txt"  # Replace with your file path
    id_list = extract_ids(empty_file_path)

    print(id_list)

    origin_dataset_file = "./data/formatted_bird_train.json"
    output_dataset_file = "./data/nonempty_formatted_bird_train.json"

    filter_json(origin_dataset_file, output_dataset_file, id_list)
    
