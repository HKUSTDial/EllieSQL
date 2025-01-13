import asyncio
from src.core.llm import LLMBase
from src.modules.schema_linking.basic_linker import BasicSchemaLinker
from src.modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from src.modules.schema_linking.skip_linker import SkipSchemaLinker
from src.modules.sql_generation.gpt_generator import GPTSQLGenerator
from src.modules.sql_generation.enhanced_generator import EnhancedSQLGenerator
from src.modules.sql_generation.ensemble_generator import EnsembleGenerator
from src.modules.post_processing.reflection import ReflectionPostProcessor
from src.modules.post_processing.feedback_based_reflection import FeedbackBasedReflectionPostProcessor
from src.modules.post_processing.skip_post_processing import SkipPostProcessor
from src.pipeline import ElephantSQLPipeline

async def main():
    # 初始化LLM和模块
    llm = LLMBase()
    
    # 创建不同的pipeline组合
    # pipeline1 = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-3.5-turbo", 
    #         temperature=0.5, 
    #         max_tokens=1000,
    #         max_retries=3
    #     ),
    #     sql_generator=GPTSQLGenerator(
    #         llm, 
    #         model="gpt-3.5-turbo", 
    #         temperature=0.5, 
    #         max_tokens=1000,
    #         max_retries=3
    #     ),
    #     post_processor=FeedbackBasedReflectionPostProcessor(
    #         llm, 
    #         model="gpt-3.5-turbo", 
    #         temperature=0.5, 
    #         max_tokens=1000,
    #         max_retries=3
    #     )
    # )
    
    pipeline2 = ElephantSQLPipeline(
        schema_linker=EnhancedSchemaLinker(
            llm, 
            model="gpt-3.5-turbo", 
            temperature=0.5, 
            max_tokens=1000,
            max_retries=10
        ),
        sql_generator=GPTSQLGenerator(
            llm, 
            model="gpt-3.5-turbo", 
            temperature=0.5, 
            max_tokens=1000,
            max_retries=10
        ),
        post_processor=SkipPostProcessor()
    )
    

    # pipeline3 = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-3.5-turbo", 
    #         temperature=0.5, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     sql_generator=EnhancedSQLGenerator(
    #         llm, 
    #         model="gpt-3.5-turbo", 
    #         temperature=0.5, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )

    # pipeline4 = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-3.5-turbo", 
    #         temperature=0.5, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     sql_generator=EnsembleGenerator(
    #         llm, 
    #         model="gpt-3.5-turbo", 
    #         temperature=0.5, 
    #         max_tokens=10000,
    #         n_candidates=2,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )

    # pipeline5 = ElephantSQLPipeline(
    #     schema_linker=SkipSchemaLinker(),
    #     sql_generator=GPTSQLGenerator(
    #         llm, 
    #         model="gpt-3.5-turbo", 
    #         temperature=0.5, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )
    
    # 运行pipeline
    await pipeline2.run_pipeline(data_file="./data/merge_dev_demo.json")

if __name__ == "__main__":
    asyncio.run(main()) 