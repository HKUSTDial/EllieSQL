import asyncio
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from .core.llm import LLMBase
from .modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from .modules.sql_generation.gpt_generator import GPTSQLGenerator
from .modules.sql_generation.enhanced_generator import EnhancedSQLGenerator
from .modules.sql_generation.online_synthesis_refiner_generator import OSRefinerSQLGenerator
from .modules.post_processing.skip_post_processing import SkipPostProcessor
from .pipeline import ElephantSQLPipeline
from .router.table_count_router import TableCountRouter
from .pipeline_factory import PipelineFactory, PipelineLevel

async def run_with_router(data_file: str, max_workers: int = 5):
    """使用路由器运行pipeline"""
    llm = LLMBase()
    pipeline_factory = PipelineFactory(llm)
    
    # 创建并注册生成器
    basic_pipeline = pipeline_factory.get_pipeline(PipelineLevel.BASIC)
    advanced_pipeline = pipeline_factory.get_pipeline(PipelineLevel.ADVANCED)
    
    # 创建路由器
    router = TableCountRouter("basic", "advanced")
    router.register_generator("basic", basic_pipeline.sql_generator)
    router.register_generator("advanced", advanced_pipeline.sql_generator)
    
    # 创建pipeline
    pipeline = ElephantSQLPipeline(
        schema_linker=basic_pipeline.schema_linker,  # 复用基础pipeline的schema linker
        sql_generator=router,  # 使用router替代单一生成器
        post_processor=SkipPostProcessor()
    )
    
    # 运行pipeline
    await pipeline.run_pipeline_parallel(
        data_file=data_file,
        max_workers=max_workers
    )

if __name__ == "__main__":
    # 1. 设置新的事件循环
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    # 2. 设置线程池大小
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=300))

    # 3. 运行与关闭
    loop.run_until_complete(
        run_with_router(
            data_file="./data/sampled_bird_demo.json",
            max_workers=25
        )
    )
    loop.close()
