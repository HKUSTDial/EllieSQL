import sqlite3
import json
import os

def find_latest_folder(directory):
    # 获取目录中的所有文件夹
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    if not folders:
        return None  # 如果没有文件夹，返回 None
    
    # 按字典序排序，最后一个是最新的
    latest_folder = sorted(folders)[-1]
    
    return latest_folder


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
# 加载JSONL文件
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的 JSON 对象
            data.append(json.loads(line.strip()))
    return data

def execute_sql(conn, sql):
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"SQL 执行错误: {e}")
        return None

def compare_sql_results(db_path, sql1, sql2):
    # 连接数据库
    conn = sqlite3.connect(db_path)
    
    try:
        # 执行第一个 SQL 语句
        result1 = execute_sql(conn, sql1)
        if result1 is None:
            print("gold SQL 执行失败!!! 请检查错误信息。")
        
        # 执行第二个 SQL 语句
        result2 = execute_sql(conn, sql2)
        if result2 is None:
            print("generated SQL 执行失败。")
        else:
            print(f"gold SQL结果: {result1} \n generated SQL结果: {result2}")
        
        # 比较结果
        if result1 == result2:
            return 1
        else:
            return 0
    
    finally:
        # 关闭数据库连接
        conn.close()


def compute_EX():
    merge_dev_demo_file = "./data/merge_dev_demo.json"
    merge_dev_demo_data = load_json(merge_dev_demo_file)

    time_path = "./results/intermediate_results/" + find_latest_folder("./results/intermediate_results/") 
    generated_sql_file = time_path + "/" + "generated_sql_results.jsonl"
    generated_sql_data = load_jsonl(generated_sql_file)


    total_cnt = 0
    correct_cnt = 0
    for item in merge_dev_demo_data:
        total_cnt+=1

        question_id = item.get("question_id")
        source = item.get("source", "")
        db_id = item.get("db_id", "")
        db_path = "./data/merged_databases/" + source +'_'+ db_id +"/"+ db_id + '.sqlite'

        print(db_path)

        gold_sql = item.get("gold_SQL")
        generated_sql = ""

        for i in generated_sql_data:
            if i.get("question_id") == question_id:
                generated_sql = i.get("generated_sql")
        correct_cnt += compare_sql_results(db_path, gold_sql, generated_sql)
        

    EX = round(correct_cnt/total_cnt, 2)
    print(f"总计问题数量：{total_cnt}, 生成sql正确执行数量{correct_cnt}, EX值{EX}")


if __name__ == "__main__":
    compute_EX()


