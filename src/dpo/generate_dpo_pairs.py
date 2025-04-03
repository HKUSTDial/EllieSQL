import json
import argparse
from ..core.utils import load_jsonl

# Read the original dataset
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Generate preference pairs
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

# Save as a JSON file
def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled_file', type=str, required=True,
                       help='Path to the dataset with labels which is to be used for DPO dataset preparation')
    parser.add_argument('--preference_file', type=str, required=True,
                       help='Path to the dataset with preference pairs which is to be used for DPO dataset preparation')
    args = parser.parse_args()

    labeled_file = args.labeled_file
    preference_file = args.preference_file


    data = load_json(labeled_file)
    preference_pairs = generate_preference_pairs(data)
    save_json(preference_pairs, preference_file)
    
    print(f"Generated {len(preference_pairs)} preference pairs, saved in {preference_file}")

if __name__ == "__main__":
    main()
