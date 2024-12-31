from typing import Dict, Any, Tuple, Optional
import re
from src.core.llm import LLMBase
from src.modules.schema_linking.base import SchemaLinkerBase
from src.modules.sql_generation.base import SQLGeneratorBase
from src.modules.post_processing.base import PostProcessorBase

class ElephantSQLPipeline:
    """ElephantSQL系统的主要pipeline"""
    
    def __init__(self,
                 schema_linker: SchemaLinkerBase,
                 sql_generator: SQLGeneratorBase,
                 post_processor: PostProcessorBase,
                 max_retries: int = 5):
        """
        初始化pipeline
        
        Args:
            schema_linker: Schema linking模块实例
            sql_generator: SQL生成模块实例
            post_processor: 后处理模块实例
            max_retries: SQL生成最大重试次数
        """
        self.schema_linker = schema_linker
        self.sql_generator = sql_generator
        self.post_processor = post_processor
        self.max_retries = max_retries
        
    def _extract_sql(self, text: str) -> Optional[str]:
        """
        从文本中提取SQL代码块
        
        Args:
            text: 包含SQL的文本
            
        Returns:
            Optional[str]: 提取出的SQL语句，如果没有找到则返回None
        """
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, text, re.DOTALL)
        return matches[0].strip() if matches else None
        
    async def _generate_sql_with_retry(self, query: str, linked_schema: Dict) -> Tuple[str, str]:
        """
        使用重试机制生成SQL
        
        Args:
            query: 用户查询
            linked_schema: 链接后的schema
            
        Returns:
            Tuple[str, str]: (原始输出, 提取的SQL)
        """
        for attempt in range(self.max_retries):
            raw_output = await self.sql_generator.generate_sql(query, linked_schema)
            sql = self._extract_sql(raw_output)
            if sql:
                return raw_output, sql
        raise ValueError(f"无法在{self.max_retries}次尝试中生成有效的SQL")
        
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
        
        # 2. SQL Generation with retry
        raw_sql_generation_output, extracted_generated_sql = await self._generate_sql_with_retry(query, linked_schema)
        
        # 3. Post Processing
        raw_postprocessing_output = await self.post_processor.process_sql(extracted_generated_sql)
        processed_sql = self._extract_sql(raw_postprocessing_output) or raw_postprocessing_output
        
        return {
            "query": query,
            "original_schema": linked_schema["original_schema"],
            "linked_schema": linked_schema["linked_schema"],
            "sql_generation_output": raw_sql_generation_output,
            "generated_sql": extracted_generated_sql,
            "postprocessing_output": raw_postprocessing_output,
            "processed_sql": processed_sql
        } 