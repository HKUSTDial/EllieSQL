import asyncio
from src.core.llm import LLMBase
from src.modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from src.modules.sql_generation.gpt_generator import GPTSQLGenerator
from src.modules.post_processing.reflection import ReflectionPostProcessor
from src.pipeline import ElephantSQLPipeline

async def main():
    # 初始化LLM和模块
    llm = LLMBase()
    
    schema_linker = EnhancedSchemaLinker(
        llm, 
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        max_tokens=1000
    )
    
    sql_generator = GPTSQLGenerator(
        llm, 
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        max_tokens=1000
    )
    
    post_processor = ReflectionPostProcessor(
        llm, 
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        max_tokens=1000
    )
    
    # 创建并运行pipeline
    pipeline = ElephantSQLPipeline(
        schema_linker=schema_linker,
        sql_generator=sql_generator,
        post_processor=post_processor
    )
    
    # 运行pipeline
    await pipeline.run_pipeline(data_file="./data/sampled_merged.json")

if __name__ == "__main__":
    asyncio.run(main()) 