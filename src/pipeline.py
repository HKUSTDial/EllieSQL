from typing import Dict, Any
from src.core.llm import LLMBase
from src.modules.schema_linking.base import SchemaLinkerBase
from src.modules.sql_generation.base import SQLGeneratorBase
from src.modules.post_processing.base import PostProcessorBase

class ElephantSQLPipeline:
    """ElephantSQL系统的主要pipeline"""
    
    def __init__(self,
                 schema_linker: SchemaLinkerBase,
                 sql_generator: SQLGeneratorBase,
                 post_processor: PostProcessorBase):
        """
        初始化pipeline
        
        Args:
            schema_linker: Schema linking模块实例
            sql_generator: SQL生成模块实例
            post_processor: 后处理模块实例
        """
        self.schema_linker = schema_linker
        self.sql_generator = sql_generator
        self.post_processor = post_processor
        
    async def process(self, 
                    query: str, 
                    database_schema: Dict) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户的自然语言查询
            database_schema: 数据库schema信息
            
        Returns:
            Dict[str, Any]: 包含处理结果的字典
        """
        # 1. Schema Linking
        linked_schema = await self.schema_linker.link_schema(query, database_schema)
        
        # 2. SQL Generation
        raw_sql = await self.sql_generator.generate_sql(query, linked_schema)
        
        # 3. Post Processing
        final_sql = await self.post_processor.process_sql(raw_sql)
        
        return {
            "query": query,
            "linked_schema": linked_schema,
            "raw_sql": raw_sql,
            "final_sql": final_sql
        } 