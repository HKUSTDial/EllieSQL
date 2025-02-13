from ..core.sql_execute import *
from ..core.utils import load_json, load_jsonl
from ..core.config import Config


def get_sql_result(db_path, sql, timeout=10):
    """
    get the ex results for a sql.
    
    Args:
        db_path: The path to the .sqlite file.
        sql: The sql.
    Returns:
        0 for sql success (nonempty result),
        1 for SQL fail, 
        2 for sql success but empty result.
    """
    # 执行第一个 SQL 语句
    result1 = execute_sql_with_timeout(db_path, sql, timeout)
    if result1 is None:
        return 1

    if result1.result is None:
        return 2
    
    return 0


def count_empty_sql_result(dataset_file: str, output_file: str):
    """
    统计一个数据集中，goldsql结果为空的对象的数量，并将该对象的id和错误类型存在txt文件里。
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
                error_msg = f"id {question_id}: gold SQL 执行失败!!!! 请检查错误信息。\n"
                print(error_msg.strip())
                f.write(error_msg)
            elif(curr_res == 2):
                gold_empty_cnt += 1
                error_msg = f"id {question_id}: gold SQL 执行成功但结果为空。\n"
                print(error_msg.strip())
                f.write(error_msg)

    print(f"统计结果：gold SQL 执行成功但结果非空: {gold_correct_cnt}，gold SQL 执行失败: {gold_fail_cnt}，gold SQL 执行成功但结果为空: {gold_empty_cnt}")


if __name__ == "__main__":
    dataset_file = "./data/formatted_bird_train.json"
    output_file = "./data/bird_train_gold_empty_count.txt"
    count_empty_sql_result(dataset_file, output_file)