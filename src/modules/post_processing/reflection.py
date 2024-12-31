from typing import Dict, List
from .base import PostProcessorBase
from ...core.llm import LLMBase
from .prompts.reflection_prompts import REFLECTION_SYSTEM, REFLECTION_USER

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
        
    async def process_sql(self, sql: str) -> str:
        """
        对生成的SQL进行自反思检查和优化
        
        Args:
            sql: 原始SQL语句
        """
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM},
            {"role": "user", "content": REFLECTION_USER.format(sql=sql)}
        ]
        
        # 调用LLM
        result = await self.llm.call_llm(
            messages, 
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        processed_sql = result["response"].strip()
        self.log_io({"original_sql": sql}, processed_sql)
        return processed_sql 