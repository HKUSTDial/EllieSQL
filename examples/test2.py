import json

def check_sql_consistency(file1_path, file2_path):

    with open(file1_path, 'r', encoding='utf-8') as file1:
        file1_data = [json.loads(line) for line in file1]
    

    with open(file2_path, 'r', encoding='utf-8') as file2:
        file2_data = [json.loads(line) for line in file2]
    

    file2_dict = {entry['query_id']: entry['output']['extracted_sql'] for entry in file2_data}
    

    inconsistencies = []
    for entry in file1_data:
        question_id = entry['query_id']
        post_input_original_sql = entry['input']['original_sql']
        
        if question_id in file2_dict:
            generated_sql = file2_dict[question_id]
            if post_input_original_sql != generated_sql:
                inconsistencies.append({
                    'question_id': question_id,
                    'post_input_original_sql': post_input_original_sql,
                    'generator_output_sql': generated_sql
                })
    
    return inconsistencies

# 使用示例
file1_path = 'results/intermediate_results/20250117_192112/SkipPostProcessor.jsonl'
file2_path = 'results\intermediate_results\\20250117_192112\VanillaRefineSQLGenerator.jsonl'
inconsistencies = check_sql_consistency(file1_path, file2_path)

if inconsistencies:
    for inconsistency in inconsistencies:
        print(f"Question ID: {inconsistency['question_id']}")
        print(f" - post_input_original_sql:\n {inconsistency['post_input_original_sql']}")
        print(f" - generator_output_sql:\n {inconsistency['generator_output_sql']}\n")
else:
    print("ok")
