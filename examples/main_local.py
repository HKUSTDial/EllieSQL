import asyncio
from src.core.llm import LLMBase
from src.modules.schema_linking.basic_linker import BasicSchemaLinker
from src.modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from src.modules.schema_linking.skip_linker import SkipSchemaLinker
from src.modules.sql_generation.gpt_generator import GPTSQLGenerator
from src.modules.sql_generation.vanilla_refiner_generator import VanillaRefineSQLGenerator
from src.modules.sql_generation.enhanced_generator import EnhancedSQLGenerator
from src.modules.sql_generation.online_synthesis_refiner_generator import OSRefinerSQLGenerator
from src.modules.sql_generation.query_plan_refiner_generator import QPRefinerSQLGenerator
from src.modules.sql_generation.dc_refiner_generator import DCRefinerSQLGenerator
from src.modules.sql_generation.chase_generator import CHASESQLGenerator
from src.modules.sql_generation.ensemble_generator import EnsembleGenerator
from src.modules.sql_generation.os_refiner_aggregation_generator import OSRefinerAggregationSQLGenerator
from src.modules.post_processing.reflection import ReflectionPostProcessor
from src.modules.post_processing.feedback_based_reflection import FeedbackBasedReflectionPostProcessor
from src.modules.post_processing.skip_post_processing import SkipPostProcessor
from src.pipeline import ElephantSQLPipeline
from concurrent.futures import ThreadPoolExecutor

async def main(backbone_model: str = 'gpt-4o-mini-2024-07-18'):
    # 初始化LLM和模块
    llm = LLMBase()
    
    pipeline_v = ElephantSQLPipeline(
        schema_linker=EnhancedSchemaLinker(
            llm, 
            model=backbone_model, # Still use same model for schema linking
            temperature=0.0, 
            max_tokens=10000,
            max_retries=10
        ),
        sql_generator=DCRefinerSQLGenerator(
            llm, 
            model='qwen2.5-coder-7b-instruct', # Use local model for SQL generation
            temperature=0.001,
            max_tokens=10000,
            max_retries=3
        ),
        post_processor=SkipPostProcessor()
    )

    # 运行pipeline，设置并行数
    await pipeline_v.run_pipeline_parallel(
        # data_file="./data/sampled_bird_dev.json", # 20% of bird dev
        data_file="./data/formatted_bird_dev.json", # 100% of bird dev
        max_workers=1
    )

if __name__ == "__main__":
    # 串行化运行
    # asyncio.run(main())

    # 并行化运行
    # 1. 设置新的事件循环
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    # 2. 设置线程池大小
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))

    # 3. 运行与关闭
    loop.run_until_complete(main(backbone_model="gpt-3.5-turbo"))
    loop.close() 