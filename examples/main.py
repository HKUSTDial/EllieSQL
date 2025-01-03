import asyncio
from src.core.llm import LLMBase
from src.modules.schema_linking.basic_linker import BasicSchemaLinker
from src.modules.sql_generation.gpt_generator import GPTSQLGenerator
from src.modules.post_processing.reflection import ReflectionPostProcessor
from src.pipeline import ElephantSQLPipeline


import json

# 加载JSON文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

async def main():
    # 初始化LLM
    llm = LLMBase()
    
    # 初始化各个模块，可以指定具体的参数
    schema_linker = BasicSchemaLinker(
        llm, 
        model="gpt-3.5-turbo-0613",
        temperature=0.5,
        max_tokens=1000
    )
    
    sql_generator = GPTSQLGenerator(
        llm, 
        model="gpt-3.5-turbo-0613",
        temperature=0.5,
        max_tokens=1000
    )
    
    post_processor = ReflectionPostProcessor(
        llm, 
        model="gpt-3.5-turbo-0613",
        temperature=0.5,
        max_tokens=1000
    )
    
    # 创建pipeline
    pipeline = ElephantSQLPipeline(
        schema_linker=schema_linker,
        sql_generator=sql_generator,
        post_processor=post_processor
    )
    

    spider_dev_schemas_file = ".\\data\\spider_dev_schemas.json"
    bird_dev_schemas_file = "./data/bird_dev_schemas.json"
    spider_dev_schemas_data = load_json(spider_dev_schemas_file)
    bird_dev_schemas_data = load_json(bird_dev_schemas_file)

    merge_devs_demo_file = "./data/merge_devs_demo.json"
    merge_devs_demo_data = load_json(merge_devs_demo_file)

    for item in merge_devs_demo_data:
        source = item.get("source", "")
        schema_file = source+"_schemas.json"
        schema_data = load_json("./data/"+schema_file)

        db_id = item.get("db_id", "")
        schema = {}
        for i in schema_data:
            if i.get("database") == db_id:
                schema = i
                break

        query = item.get("question")

        #print(schema)

    # # 示例schema
    # schema = {
    #     "tables": [
    #         {
    #             "name": "students",
    #             "columns": ["id", "name", "age", "class_id", "gender", "birthday", "email", "phone"]
    #         },
    #         {
    #             "name": "classes",
    #             "columns": ["id", "name", "teacher_id", "location"]
    #         }
    #     ]
    # }
    
    # # 示例查询
    # query = "找出所有18岁以上的学生的姓名和性别及其所在班级的名称"
    
        # 处理查询
        result = await pipeline.process(query, schema)
    
        # 打印结果
        print("最终SQL:", result["processed_sql"])
        print("result:", result)
        print("\nLLM调用统计:")
        print(f"总调用次数: {llm.metrics.total_calls}")
        print(f"每个模型调用次数: {llm.metrics.model_calls}")
        print(f"总输入tokens: {llm.metrics.total_input_tokens}")
        print(f"总输出tokens: {llm.metrics.total_output_tokens}")
        print("\n按模块统计:")
        print(llm.metrics.module_stats)
        for module, stats in llm.metrics.module_stats.items():
            print(f"\n{module}:")
            print(f"  调用次数: {stats['calls']}")
            print(f"  输入tokens: {stats['input_tokens']}")
            print(f"  输出tokens: {stats['output_tokens']}")

if __name__ == "__main__":
    asyncio.run(main()) 