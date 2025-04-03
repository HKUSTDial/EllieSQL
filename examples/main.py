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
    # Initialize LLM and modules
    llm = LLMBase()
    
    pipeline_v = ElephantSQLPipeline(
        schema_linker=EnhancedSchemaLinker(
            llm, 
            model=backbone_model, 
            temperature=0.0, 
            max_tokens=10000,
            max_retries=10
        ),
        sql_generator=GPTSQLGenerator(
            llm, 
            model=backbone_model, 
            temperature=0.0, 
            max_tokens=10000,
            max_retries=10
        ),
        post_processor=SkipPostProcessor()
    )
    
    # Run pipeline, set parallel number
    await pipeline_v.run_pipeline_parallel(
        data_file="./data/formatted_bird_dev.json",
        max_workers=100
    )

if __name__ == "__main__":
    # Run serially
    # asyncio.run(main())

    # Run in parallel
    # 1. Set new event loop
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    # 2. Set thread pool size
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=300))

    # 3. Run and close
    loop.run_until_complete(main(backbone_model="gpt-3.5-turbo"))
    loop.close() 

    # python -m examples.main