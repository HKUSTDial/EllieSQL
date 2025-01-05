import asyncio
from src.core.llm import LLMBase
from src.modules.schema_linking.basic_linker import BasicSchemaLinker
from src.modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from src.modules.sql_generation.gpt_generator import GPTSQLGenerator
from src.modules.post_processing.reflection import ReflectionPostProcessor
from src.modules.post_processing.feedback_based_reflection import FeedbackBasedReflectionPostProcessor
from src.pipeline import ElephantSQLPipeline

async def main():
    # 初始化LLM和模块
    llm = LLMBase()
    
    # 创建不同的pipeline组合
    # pipeline1 = ElephantSQLPipeline(
    #     schema_linker=BasicSchemaLinker(llm),
    #     sql_generator=GPTSQLGenerator(llm),
    #     post_processor=ReflectionPostProcessor(llm)
    # )
    
    pipeline2 = ElephantSQLPipeline(
        schema_linker=EnhancedSchemaLinker(llm),  # 使用增强版schema linker
        sql_generator=GPTSQLGenerator(llm),
        post_processor=FeedbackBasedReflectionPostProcessor(llm)  # 使用不同的后处理器
    )
    
    # 运行pipeline
    # await pipeline1.run_pipeline(data_file="./data/merge_dev_demo.json")
    await pipeline2.run_pipeline(data_file="./data/merge_dev_demo.json")

if __name__ == "__main__":
    asyncio.run(main()) 