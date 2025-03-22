import asyncio
import argparse
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

async def run_pipeline(pipeline_name: str, schema_model: str, generator_model: str, test_file: str, max_workers: int):
    """运行指定的pipeline"""
    llm = LLMBase()
    
    # 创建schema linker
    schema_linker = EnhancedSchemaLinker(
        llm, 
        model=schema_model, 
        temperature=0.0, 
        max_tokens=10000,
        max_retries=10
    )
    
    # 根据pipeline名称选择generator
    if pipeline_name == "GB":
        generator = GPTSQLGenerator(
            llm, 
            model=generator_model, 
            temperature=0.0, 
            max_tokens=10000,
            max_retries=10
        )
    elif pipeline_name == "GM":
        generator = DCRefinerSQLGenerator(
            llm, 
            model=generator_model, 
            temperature=0.0, 
            max_tokens=10000,
            max_retries=10
        )
    elif pipeline_name == "GA":
        generator = EnhancedSQLGenerator(
            llm, 
            model=generator_model, 
            temperature=0.0, 
            max_tokens=10000,
            max_retries=10
        )
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    
    # 创建pipeline
    pipeline = ElephantSQLPipeline(
        schema_linker=schema_linker,
        sql_generator=generator,
        post_processor=SkipPostProcessor()
    )
    
    # 运行pipeline
    print(f"\nRunning {pipeline_name} pipeline:")
    print(f"Schema Linker Model: {schema_model}")
    print(f"Generator Model: {generator_model}")
    
    await pipeline.run_pipeline_parallel(
        data_file=test_file,
        max_workers=max_workers
    )

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, required=True,
                       choices=['GB', 'GM', 'GA'],
                       help='Pipeline to run (GB/GM/GA)')
    parser.add_argument('--schema_model', type=str, default="gpt-3.5-turbo",
                       help='Model for schema linking')
    parser.add_argument('--generator_model', type=str, default="gpt-3.5-turbo",
                       help='Model for SQL generation')
    parser.add_argument('--test_file', type=str, default="./data/formatted_bird_dev.json",
                       help='Test data file path')
    parser.add_argument('--max_workers', type=int, default=64,
                       help='Maximum number of parallel workers')
    args = parser.parse_args()
    
    await run_pipeline(
        pipeline_name=args.pipeline,
        schema_model=args.schema_model,
        generator_model=args.generator_model,
        test_file=args.test_file,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    # 1. 设置新的事件循环
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    # 2. 设置线程池大小
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=256))

    # 3. 运行与关闭
    loop.run_until_complete(main())
    loop.close() 