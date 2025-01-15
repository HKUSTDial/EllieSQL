from typing import Dict, List
from ....core.llm import LLMBase
from ....core.sql_execute import *
from ....core.utils import load_json, load_jsonl
from ..prompts.online_synthesis_cot_prompts import SQL_GENERATION_SYSTEM, ONLINE_SYNTHESIS_PROMPT


from ..base import ModuleBase
from typing import Any, Dict, Optional, Tuple, Callable


class OnlineSynthesiser():
    """基于指定DB Schema, 在线合成该DB的NL2SQL examples"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 5000,
                module_name: str = "EnhancedSQLGenerator"):
        # super().__init__("FeedbackBasedRefiner")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.module_name = module_name

    async def generate_online_synthesis_examples(self, formatted_schema: str) -> Dict:
        """
        生成在线合成的示例
        
        Args:
            formatted_schema: 格式化的schema字符串
            
        Returns:
            Dict: 包含示例和token统计的结果字典
        """
        # 构建提示词
        online_synthesis_messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": ONLINE_SYNTHESIS_PROMPT.format(
                TARGET_DATABASE_SCHEMA=formatted_schema,
                k=2  # 或者根据需要调整
            )}
        ]
        
        # 调用 LLM 获取示例
        result = await self.llm.call_llm(
            online_synthesis_messages,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.module_name
        )
        # result = await generator.llm.call_llm(
        #     online_synthesis_messages,
        #     generator.model,
        #     temperature=generator.temperature,
        #     max_tokens=generator.max_tokens,
        #     module_name="EnhancedSQLGenerator"
        # )
        print("生成examples完成")
        return result  # 返回完整的结果字典

                
        