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
from src.modules.post_processing.reflection import ReflectionPostProcessor
from src.modules.post_processing.feedback_based_reflection import FeedbackBasedReflectionPostProcessor
from src.modules.post_processing.skip_post_processing import SkipPostProcessor
from src.pipeline import ElephantSQLPipeline
from concurrent.futures import ThreadPoolExecutor

async def main():
    # 初始化LLM和模块
    llm = LLMBase()
    
    # 创建不同的pipeline组合
    # pipeline1 = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.5, 
    #         max_tokens=1000,
    #         max_retries=3
    #     ),
    #     sql_generator=GPTSQLGenerator(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.5, 
    #         max_tokens=1000,
    #         max_retries=3
    #     ),
    #     post_processor=FeedbackBasedReflectionPostProcessor(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.5, 
    #         max_tokens=1000,
    #         max_retries=3
    #     )
    # )
    
    # pipeline_v = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     sql_generator=GPTSQLGenerator(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )
    
    # pipeline_vr = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     sql_generator=VanillaRefineSQLGenerator(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )

    pipeline_enh = ElephantSQLPipeline(
        schema_linker=EnhancedSchemaLinker(
            llm, 
            model="gpt-4o-mini-2024-07-18", 
            temperature=0.0, 
            max_tokens=10000,
            max_retries=3
        ),
        sql_generator=EnhancedSQLGenerator(
            llm, 
            model="gpt-4o-mini-2024-07-18", 
            temperature=0.0, 
            max_tokens=5000,
            max_retries=3
        ),
        post_processor=SkipPostProcessor()
    )

    # pipeline_osr = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     sql_generator=OSRefinerSQLGenerator(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )

    # pipeline_qpr = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     sql_generator=QPRefinerSQLGenerator(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=5000,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )

    # pipeline_dcr = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=10000,
    #         max_retries=3

    #     ),
    #     sql_generator=DCRefinerSQLGenerator(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=5000,
    #         max_retries=3

    #     ),
    #     post_processor=SkipPostProcessor()
    # )

    # pipeline_chase = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.0, 
    #         max_tokens=5000,
    #         max_retries=3
    #     ),
    #     sql_generator=CHASESQLGenerator(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.5, 
    #         max_tokens=5000,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )

    # pipeline4 = ElephantSQLPipeline(
    #     schema_linker=EnhancedSchemaLinker(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.5, 
    #         max_tokens=5000,
    #         max_retries=3
    #     ),
    #     sql_generator=EnsembleGenerator(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.5, 
    #         max_tokens=5000,
    #         n_candidates=2,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )

    # pipeline5 = ElephantSQLPipeline(
    #     schema_linker=SkipSchemaLinker(),
    #     sql_generator=GPTSQLGenerator(
    #         llm, 
    #         model="gpt-4o-mini-2024-07-18", 
    #         temperature=0.5, 
    #         max_tokens=10000,
    #         max_retries=3
    #     ),
    #     post_processor=SkipPostProcessor()
    # )


    
    # 运行pipeline，设置并行数
    await pipeline_enh.run_pipeline_parallel(
        # data_file="./data/merge_dev_demo.json",
        # data_file="./data/sampled_merged.json",
        # data_file="./data/sampled_bird_dev.json", # 20% of bird dev
        # data_file="./data/formatted_bird_dev.json", # 100% of bird dev
        data_file="./data/sampled_bird_demo.json",
        max_workers=10
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
    loop.set_default_executor(ThreadPoolExecutor(max_workers=300))

    # 3. 运行与关闭
    loop.run_until_complete(main())
    loop.close() 