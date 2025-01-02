from typing import Dict, Any
from datetime import datetime
from src.core.llm import LLMBase
from src.modules.schema_linking.base import SchemaLinkerBase
from src.modules.sql_generation.base import SQLGeneratorBase
from src.modules.post_processing.base import PostProcessorBase
from src.core.intermediate import IntermediateResult

class ElephantSQLPipeline:
    """ElephantSQL系统的主要pipeline"""
    
    def __init__(self,
                 schema_linker: SchemaLinkerBase,
                 sql_generator: SQLGeneratorBase,
                 post_processor: PostProcessorBase):
        self.pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 为每个模块设置pipeline_id
        schema_linker.intermediate = IntermediateResult(schema_linker.name, self.pipeline_id)
        sql_generator.intermediate = IntermediateResult(sql_generator.name, self.pipeline_id)
        post_processor.intermediate = IntermediateResult(post_processor.name, self.pipeline_id)
        
        self.schema_linker = schema_linker
        self.sql_generator = sql_generator
        self.post_processor = post_processor
        
    async def process(self, 
                    query: str, 
                    database_schema: Dict,
                    query_id: str = None) -> Dict[str, Any]:
        """处理用户查询"""
        if query_id is None:
            query_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # 1. Schema Linking with retry
        raw_linking_output, extracted_schema = await self.schema_linker.link_schema_with_retry(
            query, 
            database_schema,
            query_id=query_id
        )
        
        schema_linking_output = {
            "original_schema": database_schema,
            "linked_schema": extracted_schema
        }
        
        # 2. SQL Generation with retry
        raw_sql_output, extracted_sql = await self.sql_generator.generate_sql_with_retry(
            query,
            schema_linking_output,
            query_id=query_id
        )
        
        # 3. Post Processing with retry
        raw_postprocessing_output, processed_sql = await self.post_processor.process_sql_with_retry(
            extracted_sql,
            query_id=query_id
        )
        
        return {
            "query_id": query_id,
            "query": query,
            "raw_linking_output": raw_linking_output,
            "extracted_schema": extracted_schema,
            "raw_sql_output": raw_sql_output,
            "extracted_sql": extracted_sql,
            "raw_postprocessing_output": raw_postprocessing_output,
            "processed_sql": processed_sql
        } 