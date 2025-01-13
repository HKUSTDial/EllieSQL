from typing import Dict
from .base import PostProcessorBase
from ...core.llm import LLMBase

class SkipPostProcessor(PostProcessorBase):
    """直接返回SQL生成模块输出的后处理器，不进行任何处理，即跳过Post Processing"""
    
    def __init__(self, max_retries: int = 1):
        super().__init__("SkipPostProcessor", max_retries)
        
    async def process_sql(self, sql: str, query_id: str = None) -> str:
        """
        直接返回输入的SQL，不做任何处理
        
        Args:
            sql: 原始SQL语句
            query_id: 查询ID，用于关联中间结果
            
        Returns:
            str: 原始SQL语句
        """
        # 加载SQL生成的结果以获取原始查询
        prev_result = self.load_previous_result(query_id)
        original_query = prev_result["input"]["query"]
        
        # 保存中间结果，保持跟踪
        self.save_intermediate(
            input_data={
                "original_query": original_query,
                "original_sql": sql
            },
            output_data={
                "processed_sql": sql  # 直接使用输入的SQL
            },
            model_info={
                "model": "none",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            query_id=query_id
        )
        
        self.log_io({"original_sql": sql}, sql)
        return sql 