import json

def check_sql_equality(jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                data = json.loads(line.strip())
                original_sql = data['input']['original_sql']
                processed_sql = data['output']['processed_sql']
                
                if original_sql != processed_sql:
                    print(f"Line {line_number}: original_sql and processed_sql are different.")
                    print(f"original sql:\n{original_sql}")
                    print(f"processed sql:\n{processed_sql}")
                #else:
                    #print(f"Line {line_number}: original_sql and processed_sql are the same.")
                    
            except json.JSONDecodeError as e:
                print(f"Line {line_number}: Error decoding JSON - {e}")
            except KeyError as e:
                print(f"Line {line_number}: Missing key in JSON - {e}")
        print("ok")
# 替换为你的JSONL文件路径
jsonl_file = 'results\intermediate_results\\20250117_192112\SkipPostProcessor.jsonl'
check_sql_equality(jsonl_file)