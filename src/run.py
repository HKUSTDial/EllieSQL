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
from src.router.table_count_router import TableCountRouter
from src.router.qwen_classifier_router import QwenClassifierRouter
from src.pipeline_factory import PipelineFactory, PipelineLevel

async def main():
    # 初始化LLM
    llm = LLMBase()
    
    # 创建pipeline工厂
    factory = PipelineFactory(llm, backbone_model="gpt-3.5-turbo", temperature=0.0, max_retries=10)
    
    # 创建router
    # router = TableCountRouter()
    router = QwenClassifierRouter(
        seed=42,
        lora_path="/data/zhuyizhang/saves/Qwen2.5-0.5B-router/important/qwen_classifier_on_bird_dev_penalty/final_model_classifier"  # 可选参数
    )
    
    # 注册生成器
    router.register_generator(PipelineLevel.BASIC.value, 
                            factory.get_pipeline(PipelineLevel.BASIC).sql_generator)
    router.register_generator(PipelineLevel.INTERMEDIATE.value, 
                            factory.get_pipeline(PipelineLevel.INTERMEDIATE).sql_generator)
    router.register_generator(PipelineLevel.ADVANCED.value, 
                            factory.get_pipeline(PipelineLevel.ADVANCED).sql_generator)
    
    # 创建使用router的pipeline
    pipeline = ElephantSQLPipeline(
        schema_linker=factory.get_pipeline(PipelineLevel.BASIC).schema_linker, # 复用基础pipeline的schema linker
        sql_generator=router, # 使用router代替特定的generator
        post_processor=SkipPostProcessor()
    )
    
    # 运行pipeline
    await pipeline.run_pipeline_parallel(
        data_file="./data/formatted_bird_dev.json",
        max_workers=200
    )

if __name__ == "__main__":
    # 1. 设置新的事件循环
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    # 2. 设置线程池大小
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=600))

    # 3. 运行与关闭
    loop.run_until_complete(main())
    loop.close()
