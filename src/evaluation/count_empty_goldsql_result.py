from ..core.sql_execute import *
from ..core.utils import load_json, load_jsonl
from ..core.config import Config


def get_sql_result(db_path, sql, timeout=10):
    """
    Get the EX results for a SQL query.
    
    :param db_path: The path to the .sqlite file.
    :param sql: The sql.
    :return:
        0 for sql success (nonempty result),
        1 for SQL fail, 
        2 for sql success but empty result.
    """
    # Execute the first SQL statement
    result1 = execute_sql_with_timeout(db_path, sql, timeout)
    if result1 is None:
        return 1

    if result1.result is None:
        return 2
    
    return 0


def count_empty_sql_result(dataset_file: str, output_file: str):
    """
    Count the number of objects in a dataset where the goldsql result is empty, and save the object id and error type to a txt file.
    """
    dataset_data = load_json(dataset_file)

    gold_correct_cnt = 0
    gold_fail_cnt = 0
    gold_empty_cnt = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset_data:
            question_id = item.get("question_id")
            source = item.get("source", "")
            db_id = item.get("db_id", "")
            db_folder = f"{source}_{db_id}"
            db_file = f"{db_id}.sqlite"
            db_path = str(Config().database_dir / db_folder / db_file)

            gold_sql = item.get("gold_SQL")

            curr_res = get_sql_result(db_path, gold_sql)
            if(curr_res == 0):
                gold_correct_cnt += 1
            elif(curr_res == 1):
                gold_fail_cnt += 1
                error_msg = f"id {question_id}: gold SQL execution failed!!!! Please check the error information.\n"
                print(error_msg.strip())
                f.write(error_msg)
            elif(curr_res == 2):
                gold_empty_cnt += 1
                error_msg = f"id {question_id}: gold SQL execution success but the result is empty.\n"
                print(error_msg.strip())
                f.write(error_msg)

    print(f"Statistics: gold SQL execution success but the result is not empty: {gold_correct_cnt}, gold SQL execution failed: {gold_fail_cnt}, gold SQL execution success but the result is empty: {gold_empty_cnt}")


if __name__ == "__main__":
    dataset_file = "./data/formatted_bird_train.json"
    output_file = "./data/bird_train_gold_empty_count.txt"
    count_empty_sql_result(dataset_file, output_file)