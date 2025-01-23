import asyncio
from typing import Dict, List
from .core.llm import LLMBase
from .modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from .modules.sql_generation.gpt_generator import GPTSQLGenerator
from .modules.sql_generation.enhanced_generator import EnhancedSQLGenerator
from .modules.sql_generation.online_synthesis_refiner_generator import OSRefinerSQLGenerator
from .modules.post_processing.skip_post_processing import SkipPostProcessor
from .pipeline import ElephantSQLPipeline
from .router.table_count_router import TableCountRouter

async def run_with_router(data_file: str, max_workers: int = 5):
    """使用路由器运行pipeline"""
    
    # 初始化LLM
    llm = LLMBase()
    
    
    
    # 创建并注册生成器
    vanilla_generator = GPTSQLGenerator(
        llm, 
        model="gpt-4o-mini-2024-07-18",
        # model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=10000
    )

    osr_generator = OSRefinerSQLGenerator(
        llm,
        model="gpt-4o-2024-08-06",
        temperature=0.0,
        max_tokens=10000
    )
    
    # enhanced_generator = EnhancedSQLGenerator(
    #     llm,
    #     model="gpt-3.5-turbo",
    #     temperature=0.0,
    #     max_tokens=5000
    # )
    # 创建路由器
    router = TableCountRouter("vanilla","osr")
    router.register_generator("vanilla", vanilla_generator)
    router.register_generator("osr", osr_generator)
    # router.register_generator("enhanced", enhanced_generator)
    
    # 创建pipeline
    pipeline = ElephantSQLPipeline(
        schema_linker=EnhancedSchemaLinker(
            llm,
            model="gpt-4o-2024-08-06",
            temperature=0.0,
            max_tokens=10000
        ),
        sql_generator=router,  # 使用router替代单一生成器
        post_processor=SkipPostProcessor()
    )
    
    # 运行pipeline
    await pipeline.run_pipeline_parallel(
        data_file=data_file,
        max_workers=max_workers
    )

if __name__ == "__main__":
    # 设置事件循环
    loop = asyncio.get_event_loop()
    
    # 运行pipeline
    loop.run_until_complete(
        run_with_router(
            data_file="./data/sampled_merged.json",
            max_workers=25
        )
    )
    
    loop.close()
