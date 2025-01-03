import json
import asyncio
from src.core.llm import LLMBase
from src.modules.schema_linking.basic_linker import BasicSchemaLinker
from src.modules.sql_generation.gpt_generator import GPTSQLGenerator
from src.modules.post_processing.reflection import ReflectionPostProcessor
from src.pipeline import ElephantSQLPipeline


# 加载JSON文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

async def main():
    # 初始化LLM和模块
    llm = LLMBase()
    
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

    # 加载数据
    spider_dev_schemas_file = "./data/spider_dev_schemas.json"
    bird_dev_schemas_file = "./data/bird_dev_schemas.json"
    spider_dev_schemas_data = load_json(spider_dev_schemas_file)
    bird_dev_schemas_data = load_json(bird_dev_schemas_file)

    # 加载合并的demo数据
    merge_devs_demo_file = "./data/merge_dev_demo.json"
    merge_devs_demo_data = load_json(merge_devs_demo_file)

    for item in merge_devs_demo_data:
        # 获取问题ID
        question_id = item.get("question_id", "")  # 使用原始问题ID
        if not question_id:
            print("警告：问题缺少ID")
            continue
            
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
        
        print(f"\n处理问题 {question_id}:")
        print(f"数据库: {db_id}")
        print(f"问题: {query}")
    
        # 处理查询，使用原始问题ID和来源
        result = await pipeline.process(
            query=query, 
            database_schema=schema, 
            query_id=question_id,
            source=source
        )
    
        # 打印结果
        print("\n生成的SQL:", result["processed_sql"])
        print("\nAPI调用统计:")
        print(f"总调用次数: {llm.metrics.total_calls}")
        print(f"每个模型调用次数: {llm.metrics.model_calls}")
        print(f"总输入tokens: {llm.metrics.total_input_tokens}")
        print(f"总输出tokens: {llm.metrics.total_output_tokens}")
        print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(main()) 