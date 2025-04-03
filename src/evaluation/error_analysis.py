
from ..core.utils import load_json, load_jsonl, TextExtractor

from .compute_ex import execute_sql_with_timeout
from ..core.sql_execute import *

import asyncio
import json
from tqdm import tqdm
from ..core.llm import LLMBase
from ..core.config import Config
from .error_analysis_prompt import ERROR_ANALYSIS_SYSTEM, ERROR_ANALYSIS_USER
import argparse



# 1. Load the error results file
def update_error_value(file_path):
    """
    Traverse the specified JSON Lines file, add a key "error_value" to each object, with a default value of 1,
    and write the updated objects back to the original file.

    :param file_path: The path to the JSON Lines file.
    """
    # Read all lines from the file
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Traverse each line, update the JSON object
    updated_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        obj = json.loads(line)
        obj["error_value"] = 1  # Add the "error_value" key, with a default value of 1
        updated_lines.append(json.dumps(obj, ensure_ascii=False))
    
    # Write the updated objects back to the original file
    with open(file_path, "w", encoding="utf-8") as f:
        for line in updated_lines:
            f.write(line + "\n")


# 2.1 Check if there is a schema linking error
def check_sl_error(golden_file_bird_dev, error_sl_file, error_results_file):

    # Read all JSON objects from the schema file
    with open(error_sl_file, 'r', encoding='utf-8') as f:
        schema_results = [json.loads(line) for line in f if line.strip()]
    
    # Read all JSON objects from the sql file
    with open(error_results_file, 'r', encoding='utf-8') as f:
        sql_results = [json.loads(line) for line in f if line.strip()]

    # Construct a dictionary for quick lookup of sql objects by question_id
    sql_dict = {obj["question_id"]: obj for obj in sql_results}

    error_cnt = 0

    # Traverse each object in the schema file
    for schema_obj in schema_results:
        # Call the existing processing function
        result = check_sl_function(golden_file_bird_dev, schema_obj)
        # If the return value is 1, update
        if result == 1:
            query_id = schema_obj.get("query_id")
            if query_id in sql_dict:
                # Update error_value (multiply by 2)
                if sql_dict[query_id]["error_value"] == 1:
                    error_cnt += 1
                    sql_dict[query_id]["error_value"] *= 2
    
    # Update sql_results list (if you need to keep the original order)
    updated_sql_results = []
    for obj in sql_results:
        # Get the updated object from the dictionary
        updated_sql_results.append(sql_dict[obj["question_id"]])
    
    # Write the updated sql objects back to the file, each object on a line JSON
    with open(error_results_file, 'w', encoding='utf-8') as f:
        for obj in updated_sql_results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Schema linking errors: {error_cnt}个")


def check_sl_function(golden_file_bird_dev, schema_obj):
    golden_data_bird_dev = load_json(golden_file_bird_dev)
    dev_links = set(
        (table["table"].lower(), col.lower()) 
        for table in schema_obj["tables"] 
        for col in table["columns"]
        if col is not None  # Filter out None values
    )

    query_id = schema_obj["query_id"]
    gold_query_id = -1

    golden_data = golden_data_bird_dev
    gold_query_id = query_id - 3182

    if gold_query_id < 0:
        print("gold_query_id < 0, an exception occurred")
        return 1
    
    # Find the corresponding question in golden_data
    for golden in golden_data:
        if golden["id"] == gold_query_id:
            # The actual schema element set
            golden_links = set(
                (table["table"].lower(), col.lower()) 
                for table in golden["tables"] 
                for col in table["columns"]
            )
            # SRR 计算
            if golden_links.issubset(dev_links):  # If golden_links ⊆ dev_links
                return 0
            
            return 1
            break  # Stop after finding the corresponding golden data


# 2.2 Check if there is a syntax error
def check_valid_error(merge_dev_demo_file, error_results_path):
    """
    Traverse each JSON object in the file. If the object's "error_value" is 1,
    call the existing processing function. If the processing function returns 3,
    multiply "error_value" by 3, and finally save all objects back to the original file
    (assuming each line is a JSON object).
    """
    updated_objects = []
    valid_cnt = 0

    # Read each line of the file and parse it as a JSON object
    with open(error_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue

            # If error_value is 1, call the existing processing function
            if obj.get("error_value") == 1:
                qid =obj["question_id"]
                db_path = ""
                with open(merge_dev_demo_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if item.get("question_id") == qid:
                            source = item.get("source", "")
                            db_id = item.get("db_id", "")
                            db_folder = f"{source}_{db_id}"
                            db_file = f"{db_id}.sqlite"
                            db_path = str(Config().database_dir / db_folder / db_file)
                            break
                result = check_sql_run(db_path, obj["generated_sql"])
                if result == 3:
                    valid_cnt += 1
                    # Multiply error_value by 3
                    obj["error_value"] = obj["error_value"] * 3
            updated_objects.append(obj)

    # Write the updated objects back to the original file, each object on a line JSON
    with open(error_results_path, 'w', encoding='utf-8') as f:
        for obj in updated_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"Syntax errors: {valid_cnt}个")


def check_sql_run(db_path, sql2, timeout=10):
    """
    Check whether sql can run.
    :param db_path: The path to the .sqlite file.
    :param sql2: The generated sql.
    :return: 1 for can run, 3 for not runnable.
    """
    # Execute the second SQL statement
    result2 = execute_sql_with_timeout(db_path, sql2, timeout)
    if result2 is None:
        # print("generated SQL did not produce any result object.")
        return 3

    if result2.result_type == SQLExecutionResultType.TIMEOUT:
        # print(f"generated SQL run timeout. Error information: {result2.error_message}")
        return 3
    if result2.result_type == SQLExecutionResultType.ERROR:
        # print(f"generated SQL run error. Error information: {result2.error_message}")
        return 3

    if result2.result is None:
        # print("generated SQL executed successfully but the result is empty.")
        return 3
    
    # Compare the results
    return 1


async def check_other_error(golden_file_bird_dev, merge_dev_demo_file, error_results_path, model, temperature, max_tokens, name):
    """
    Process error analysis concurrently, with a maximum of 10 concurrent tasks running at a time
    """
    llm = LLMBase()
    extractor = TextExtractor()

    # Count list, index corresponding to each category
    my_list = [-1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, 0]

    # Read the development set and schema data
    with open(merge_dev_demo_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    with open(golden_file_bird_dev, 'r', encoding='utf-8') as f:
        golden_data_bird_dev = json.load(f)

    # Read all non-empty lines from the error_results file
    with open(error_results_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    updated_objects = []

    # Define a concurrency control semaphore, limit the concurrency to 10
    semaphore = asyncio.Semaphore(20)

    async def process_obj(obj):
        async with semaphore:
            # Process only records with error_value 1
            if obj.get("error_value") == 1:
                oid = obj["question_id"]
                question = ""
                gold_sql = ""
                # Find the corresponding record in the development set data
                for item in dev_data:
                    if item.get("question_id") == oid:
                        question = item.get("question", "")
                        gold_sql = item.get("gold_SQL")
                        break

                # Find the corresponding schema in the golden data (note: here we assume that id and question_id have a certain offset)
                gold_schema = None
                for i in golden_data_bird_dev:
                    if i.get("id") == oid - 3182:
                        gold_schema = i.get("tables")
                        break

                # Construct the prompt for calling the llm
                check_error_prompt = [
                    {"role": "system", "content": ERROR_ANALYSIS_SYSTEM},
                    {"role": "user", "content": ERROR_ANALYSIS_USER.format(
                        question=question,
                        generated_sql=obj["generated_sql"],
                        golden_sql=gold_sql,
                        golden_schema=gold_schema
                    )}
                ]

                # Asynchronously call the llm interface
                check_error_result = await llm.call_llm(
                    check_error_prompt,
                    model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    module_name=name
                )

                # Extract error classification
                category_list = extractor.extract_sub_questions(check_error_result['response'])
                try:
                    category = int(category_list[0])
                except (IndexError, ValueError):
                    print("Failed to extract error classification")
                    print(check_error_result)
                    return obj

                if category not in (5, 7, 11, 13):
                    print(f"Error classification error {category}")
                else:
                    my_list[category] += 1
                    obj["error_value"] *= category
            return obj

    # Create a task for each JSON object
    tasks = [asyncio.create_task(process_obj(json.loads(line))) for line in lines]

    # Use tqdm to display progress and wait for all tasks to complete
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing analysis progress"):
        updated_obj = await future
        updated_objects.append(updated_obj)

    # Write all updated objects back to the file
    with open(error_results_path, 'w', encoding='utf-8') as f:
        for obj in updated_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print("Error analysis classification completed")
    print(f"JOIN related errors {my_list[5]}个")
    print(f"GROUP BY related errors {my_list[7]}个")
    print(f"Nested queries and set operation errors {my_list[11]}个")
    print(f"Other errors {my_list[13]}个")



if __name__ == "__main__":
    """
    Evaluate steps:
    1. extract error sql results (extract_error_results.py)
    2. extract corresponding linked schema (extract_error_schema.py)
    3. do error analysis (error_analysis.py)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_sql_path', type=str, required=True,
                       help='the error_sql_results.jsonl file')
    parser.add_argument('--merge_dev_demo_file', type=str, required=True,
                       help='the nl2sql data file')
    parser.add_argument('--error_schema_results', type=str, required=True,
                       help='the output path for generated error_schema_results.jsonl')
    args = parser.parse_args()

    error_sql_path=args.error_sql_path
    error_schema_results=args.error_schema_results
    merge_dev_demo_file=args.merge_dev_demo_file

    update_error_value(error_sql_path)

    # 2.2 Check if there is a syntax error
    check_valid_error(merge_dev_demo_file, error_sql_path)

    # 2.1 Check if there is a schema linking error
    # error_schema_results = "results/raw_results/qwen_classifier_sft/error_schema_results.jsonl"
    golden_file_bird_dev = str(Config().gold_schema_linking_dir / "bird_dev_gold_schema_linking.json")
    check_sl_error(golden_file_bird_dev, error_schema_results, error_sql_path)


    # 2.3 Check if there is any other error
    model = "gpt-4o-mini-2024-07-18"
    temperature = 0.0
    max_tokens = 10000
    name = "error_checker"
    golden_file_bird_dev = str(Config().gold_schema_linking_dir / "bird_dev_gold_schema_linking.json")
    asyncio.run(check_other_error(golden_file_bird_dev, merge_dev_demo_file, error_sql_path, model, temperature, max_tokens, name))

    
