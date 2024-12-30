from typing import Dict, List
from .base import PostProcessorBase
from ...core.llm import LLMBase

class ReflectionPostProcessor(PostProcessorBase):
    """使用自反思机制的SQL后处理器"""
    
    def __init__(self, llm: LLMBase):
        self.llm = llm
        
    async def process_sql(self, sql: str) -> str:
        """
        对生成的SQL进行自反思检查和优化
        
        Args:
            sql: 原始SQL语句
        """
        messages = [
            {"role": "system", "content": """你是一个SQL专家，负责检查和优化SQL语句。
检查以下几点：
1. SQL语法是否正确
2. 表的连接条件是否合理
3. WHERE子句的条件是否合适
4. 是否可以优化性能

如果发现问题，直接返回修正后的SQL。如果没有问题，返回原SQL。
只返回SQL语句，不要其他解释。"""},
            {"role": "user", "content": f"请检查和优化以下SQL:\n{sql}"}
        ]
        
        # 调用LLM
        result = await self.llm.call_llm(messages, "gpt4")
        return result["response"].strip() 