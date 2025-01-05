from typing import Dict, List
from .base import PostProcessorBase
from ...core.llm import LLMBase
from .prompts.reflection_prompts import REFLECTION_SYSTEM, REFLECTION_USER
import re

class ReflectionPostProcessor(PostProcessorBase):
    """使用自反思机制的SQL后处理器"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 1000):
        super().__init__("ReflectionPostProcessor")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def process_sql(self, sql: str, query_id: str) -> str:
        """
        对生成的SQL进行自反思检查和优化
        
        Args:
            sql: 原始SQL语句
            query_id: 查询ID，用于关联中间结果
        """
        # 加载SQL生成的结果
        prev_result = self.load_previous(query_id, "GPTSQLGenerator")
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
        
        # 保存中间结果
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
        