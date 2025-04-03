from typing import Dict, List
from .base import PostProcessorBase
from ...core.llm import LLMBase
from .prompts.reflection_prompts import REFLECTION_SYSTEM, REFLECTION_USER
import re

class ReflectionPostProcessor(PostProcessorBase):
    """SQL post-processor using self-reflection mechanism"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 1000,
                max_retries: int = 3):
        super().__init__("ReflectionPostProcessor", max_retries)
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def process_sql(self, sql: str, query_id: str = None) -> str:
        """
        Run, self-reflect, and optimize the generated SQL
        
        :param sql: The original SQL statement
        :param query_id: The query ID, used to associate intermediate results
        """
            
        # Load the SQL generated result
        prev_result = self.load_previous_result(query_id)
        original_query = prev_result["input"]["query"]
        
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM},
            {"role": "user", "content": REFLECTION_USER.format(sql=sql)}
        ]
        
        result = await self.llm.call_llm(
            messages, 
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        raw_output = result["response"]
        processed_sql = self.extractor.extract_sql(raw_output)
        
        # Save the intermediate result
        self.save_intermediate(
            input_data={
                "original_query": original_query,
                "original_sql": sql
            },
            output_data={
                "raw_output": raw_output,
                "processed_sql": processed_sql
            },
            model_info={
                "model": self.model,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"]
            },
            query_id=query_id
        )
        
        self.log_io({"original_sql": sql, "messages": messages}, processed_sql)
        return raw_output
        