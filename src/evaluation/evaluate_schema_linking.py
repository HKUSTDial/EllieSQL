import json
from ..core.utils import load_json, load_jsonl
from ..core.config import Config
import sqlite3
import os
from pathlib import Path

def calculate_precision(golden_files, dev_file):
    # Load file content
    golden_data_spider_dev = load_json(golden_files[0])
    golden_data_spider_test = load_json(golden_files[1])
    golden_data_bird_dev = load_json(golden_files[2])
    
    dev_data = load_jsonl(dev_file)
    
    correct_links = 0  # Number of correct table/column predictions
    total_predictions = 0  # Total number of table/column predictions

    # Iterate over each question
    cnt = 0

    for dev in (dev_data):
        # Get the tables and columns of Dev
        dev_links = set(
            (table["table"].lower(), col.lower()) 
            for table in dev["tables"] 
            for col in table["columns"]
            if col is not None  # Add empty value check
        )

        query_id = dev["query_id"]
        gold_query_id = -1

        if query_id <= 1034:
            golden_data = golden_data_spider_dev
            gold_query_id = query_id
        elif query_id <= 3181:
            golden_data = golden_data_spider_test
            gold_query_id = query_id - 1035
        else: # ( <= 4715)
            golden_data = golden_data_bird_dev
            gold_query_id = query_id - 3182

        if(gold_query_id < 0):
            print("gold_query_id < 0, an exception occurred")

        # Iterate over golden_data to find the corresponding object through gold_query_id
        for golden in golden_data:
            if golden["id"] == gold_query_id:
                # is_count_star = "count(*)" in golden["gold_sql"].lower()
                golden_links = set()
                for table in golden["tables"]:
                    table_name = table["table"].lower()
                    #if not is_count_star or table["columns"]:  # 仅当没有 COUNT(*) 或列非空时才记录
                    golden_links.update((table_name, col.lower()) for col in table["columns"])
                        # if COUNT(*), all dev_links are considered correct
                # if is_count_star:
                #     correct_links += len(dev_links)
                # else: # Count the correct predictions and total predictions
                correct_links += len(golden_links & dev_links)  # Number of intersections
                total_predictions += len(dev_links)  # Number of Dev predictions
                
                cnt+=1
                if(cnt %100 == 34) :
                    print(f'Processed {cnt} questions, correct links {correct_links}, total prediction links {total_predictions}')
                break
    # Calculate precision
    precision = correct_links / total_predictions if total_predictions > 0 else 0
    return precision    


def calculate_recall(golden_file, dev_file):
    # Load file content
    golden_data_spider_dev = load_json(golden_files[0])
    golden_data_spider_test = load_json(golden_files[1])
    golden_data_bird_dev = load_json(golden_files[2])
    
    dev_data = load_jsonl(dev_file)
    
    correct_links = 0  # the number of correct table/column predictions
    total_golden_links = 0  # the total number of table/column related to the actual answer

    # Iterate over each question
    cnt = 0

    for dev in (dev_data):
        # Get the tables and columns of Dev
        dev_links = set(
            (table["table"].lower(), col.lower()) 
            for table in dev["tables"] 
            for col in table["columns"]
            if col is not None  # Add empty value check
        )

        query_id = dev["query_id"]
        gold_query_id = -1

        if query_id <= 1034:
            golden_data = golden_data_spider_dev
            gold_query_id = query_id
        elif query_id <= 3181:
            golden_data = golden_data_spider_test
            gold_query_id = query_id - 1035
        else: # ( <= 4715)
            golden_data = golden_data_bird_dev
            gold_query_id = query_id - 3182

        if(gold_query_id < 0):
            print("gold_query_id < 0, an exception occurred")

        # Iterate over golden_data to find the corresponding object through gold_query_id
        for golden in golden_data:
            if golden["id"] == gold_query_id:
                # Check if COUNT(*) exists
                # is_count_star = "count(*)" in golden["gold_sql"].lower()
                # Get the tables and columns of Golden, and standardize to lowercase
                golden_links = set()
                for table in golden["tables"]:
                    table_name = table["table"].lower()
                    #if not is_count_star or table["columns"]:  # Only record when there is no COUNT(*) or the column is not empty
                    golden_links.update((table_name, col.lower()) for col in table["columns"])

                # If COUNT(*), only consider the table level
                # if is_count_star:
                #     correct_links += len(dev_links)  # All dev_links are considered correct
                #     total_golden_links += len(dev_links)  # Consider the same number of columns in golden
                # else:
                # Count the correct predictions and standard answers
                correct_links += len(golden_links & dev_links)  # Number of intersections
                total_golden_links += len(golden_links)  # Number of Golden

                cnt+=1
                if(cnt %100 == 34) :
                    print(f'Processed {cnt} questions, correct links {correct_links}, total golden links {total_golden_links}')
                break
    # Calculate recall
    recall = correct_links / total_golden_links if total_golden_links > 0 else 0
    return recall


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0  # Avoid division by zero
    return 2 * (precision * recall) / (precision + recall)


def count_origin_columns(dev_file):
    dev_data = load_json(dev_file)

    total_columns = 0

    for item in dev_data:
        question_id = item.get("question_id")
        source = item.get("source", "")
        db_id = item.get("db_id", "")
        db_folder = f"{source}_{db_id}"
        db_file = f"{db_id}.sqlite"
        db_path = str(Config().database_dir / db_folder / db_file)
        total_columns += count_total_columns(db_path)

    return total_columns


def count_total_columns(db_path):
    """
    Count the total number of columns in all tables of the SQLite database.
    :param db_path: The path to the SQLite database file
    :return: The total number of columns in the database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_columns = 0
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        columns = cursor.fetchall()
        total_columns += len(columns)
    
    conn.close()
    return total_columns


def calculate_nsr_srr(golden_files, dev_file):
    # Load file content
    golden_data_spider_dev = load_json(golden_files[0])
    golden_data_spider_test = load_json(golden_files[1])
    golden_data_bird_dev = load_json(golden_files[2])
    
    dev_data = load_jsonl(dev_file)
    
    total_correct_links = 0  # the number of correct table/column predictions (for NSR)
    total_golden_links = 0  # the total number of actual schema elements (for NSR)
    total_strict_correct = 0  # the number of strict recall successes (for SRR)
    total_questions = len(dev_data)  # the total number of questions (for SRR)

    # Iterate over each question
    for dev in dev_data:
        # Get the tables and columns of Dev
        dev_links = set(
            (table["table"].lower(), col.lower()) 
            for table in dev["tables"] 
            for col in table["columns"]
            if col is not None  # Filter out None values
        )

        query_id = dev["query_id"]
        gold_query_id = -1

        if query_id <= 1034:
            golden_data = golden_data_spider_dev
            gold_query_id = query_id
        elif query_id <= 3181:
            golden_data = golden_data_spider_test
            gold_query_id = query_id - 1035
        else:  # query_id <= 4715
            golden_data = golden_data_bird_dev
            gold_query_id = query_id - 3182

        if gold_query_id < 0:
            print("gold_query_id < 0, an exception occurred")
            continue

        # Find the corresponding question in golden_data
        for golden in golden_data:
            if golden["id"] == gold_query_id:
                # The actual schema element set
                golden_links = set(
                    (table["table"].lower(), col.lower()) 
                    for table in golden["tables"] 
                    for col in table["columns"]
                )

                # NSR calculation
                total_correct_links += len(golden_links & dev_links)  # Intersection
                total_golden_links += len(golden_links)  # Number of standard answers

                # SRR calculation
                if golden_links.issubset(dev_links):  # If golden_links ⊆ dev_links
                    total_strict_correct += 1

                break  # Stop after finding the corresponding golden data

    # Calculate NSR
    nsr = total_correct_links / total_golden_links if total_golden_links > 0 else 0

    # Calculate SRR
    srr = total_strict_correct / total_questions if total_questions > 0 else 0

    return nsr, srr



if __name__ == '__main__':
    golden_file_spider_dev = str(Config().gold_schema_linking_dir / "spider_dev_gold_schema_linking.json")
    golden_file_spider_test = str(Config().gold_schema_linking_dir / "spider_test_gold_schema_linking.json")
    golden_file_bird_dev = str(Config().gold_schema_linking_dir / "bird_dev_gold_schema_linking.json")
    

    golden_files = []
    golden_files.append(golden_file_spider_dev)
    golden_files.append(golden_file_spider_test)
    golden_files.append(golden_file_bird_dev)
    
    # dev_file = str(Config().results_dir / "raw_results/knn_20250212_211131/linked_schema_results.jsonl")
    dev_file = str(Config().results_dir / "raw_results/knn_20250212_211131/error_schema_results.jsonl")

    # dev_file_count_origin = "./data/sampled_merged.json"
    dev_file_count_origin = "data/formatted_bird_dev.json"


    precision = calculate_precision(golden_files, dev_file)
    recall = calculate_recall(golden_files, dev_file)
    print(f"Schema Linking Precision: {precision:.4f}")
    print(f"Schema Linking Recall: {recall:.4f}")

    f1_score = calculate_f1(precision, recall)
    print(f"F1 Score: {f1_score:.4f}")

    origin_columns = count_origin_columns(dev_file_count_origin)
    print(f"The number of columns in the database corresponding to each question is {origin_columns}")

    nsr, srr = calculate_nsr_srr(golden_files, dev_file)

    print(f"nsr(non-strict recall rate): {nsr:.4f}")
    print(f"srr(strict recall rate): {srr:.4f}")



