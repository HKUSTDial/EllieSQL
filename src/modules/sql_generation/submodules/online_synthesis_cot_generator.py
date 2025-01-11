from typing import Dict, List
from ..base import SQLGeneratorBase
from ....core.llm import LLMBase
from ..prompts.online_synthesis_cot_prompts import SQL_GENERATION_SYSTEM, ONLINE_SYNTHESIS_PROMPT
import re

class OnlineSynthesis(SQLGeneratorBase):
    """使用GPT模型生成SQL的实现"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 5000):
        super().__init__("OnlineSynthesis")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def generate_examples(self, schema_linking_output: Dict, query_id: str, k: int=3) -> str:
        """生成 nl2sql example"""
        # 加载schema linking的结果
        if not schema_linking_output:
            prev_result = self.load_previous_result(query_id)
            formatted_schema = prev_result["output"]["formatted_linked_schema"]
        else:
            formatted_schema = schema_linking_output.get("formatted_linked_schema")
            if not formatted_schema:  # 如果直接传入的结果中没有格式化的schema
                prev_result = self.load_previous_result(query_id)
                formatted_schema = prev_result["output"]["formatted_linked_schema"]
        

        # 构建提示词
        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": ONLINE_SYNTHESIS_PROMPT.format(
                TARGET_DATABASE_SCHEMA = formatted_schema,
                k = k
            )}
        ]
        
        # print('\n'+messages[1]['content']+'\n')
        result = await self.llm.call_llm(
            messages,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        return result["response"]


