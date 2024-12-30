import asyncio
from src.core.llm import LLMBase
from src.modules.schema_linking.basic_linker import BasicSchemaLinker
from src.modules.sql_generation.gpt4_generator import GPT4SQLGenerator
from src.modules.post_processing.reflection import ReflectionPostProcessor
from src.pipeline import ElephantSQLPipeline

async def main():
    # 初始化LLM
    llm = LLMBase()
    
    # 初始化各个模块
    schema_linker = BasicSchemaLinker(llm)
    sql_generator = GPT4SQLGenerator(llm)
    post_processor = ReflectionPostProcessor(llm)
    
    # 创建pipeline
    pipeline = ElephantSQLPipeline(
        schema_linker=schema_linker,
        sql_generator=sql_generator,
        post_processor=post_processor
    )
    
    # 示例schema
    schema = {
        "tables": [
            {
                "name": "students",
                "columns": ["id", "name", "age", "class_id"]
            },
            {
                "name": "classes",
                "columns": ["id", "name", "teacher_id"]
            }
        ]
    }
    
    # 示例查询
    query = "找出所有18岁以上的学生及其所在班级的名称"
    
    # 处理查询
    result = await pipeline.process(query, schema)
    
    # 打印结果
    print("最终SQL:", result["final_sql"])
    print("\nLLM调用统计:")
    print(f"总调用次数: {llm.metrics.total_calls}")
    print(f"每个模型调用次数: {llm.metrics.model_calls}")
    print(f"总输入tokens: {llm.metrics.total_input_tokens}")
    print(f"总输出tokens: {llm.metrics.total_output_tokens}")
    print(f"每个模型token使用情况: {llm.metrics.model_tokens}")

if __name__ == "__main__":
    asyncio.run(main()) 