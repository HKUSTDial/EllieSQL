from typing import Dict
from .base import PostProcessorBase
from ...core.llm import LLMBase

class SkipPostProcessor(PostProcessorBase):
    """SQL post-processor that returns the output from the SQL generation module, without any processing,即跳过Post Processing"""
    
    def __init__(self, max_retries: int = 1):
        super().__init__("SkipPostProcessor", max_retries)
        
    async def process_sql(self, sql: str, query_id: str = None) -> str:
        """
        Return the input SQL directly, without any processing
        
        :param sql: The original SQL statement
        :param query_id: The query ID, used to associate intermediate results
            
        :return: The original SQL statement
        """
        # Load the SQL generated result to get the original query
        prev_result = self.load_previous_result(query_id)
        original_query = prev_result["input"]["query"]
        
        # Save the intermediate result, keeping track
        self.save_intermediate(
            input_data={
                "original_query": original_query,
                "original_sql": sql
            },
            output_data={
                "processed_sql": sql  # Use the input SQL directly
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