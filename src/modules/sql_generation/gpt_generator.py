from typing import Dict, List
from .base import SQLGeneratorBase
from ...core.llm import LLMBase
from .prompts.gpt_prompts import SQL_GENERATION_SYSTEM, SQL_GENERATION_USER

class GPTSQLGenerator(SQLGeneratorBase):
    """使用GPT模型生成SQL的实现"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 1000):
        super().__init__("GPTSQLGenerator")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def generate_sql(self, query: str, linked_schema: Dict) -> str:
        """
        使用GPT模型生成SQL
        
        Args:
            query: 用户查询
            linked_schema: Schema Linking的结果
        """
        # 构建提示词
        # schema_str = self._format_schema(linked_schema["original_schema"])
        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": SQL_GENERATION_USER.format(
                # schema_str=schema_str,
                tables=', '.join(linked_schema['linked_schema']['tables']),
                columns=', '.join(linked_schema['linked_schema']['columns']),
                query=query
            )}
        ]
        
        # 调用LLM
        result = await self.llm.call_llm(
            messages, 
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        sql = result["response"].strip()
        self.log_io(input_data={"query": query, 
                                "original_schema": linked_schema["original_schema"], 
                                "linked_schema": linked_schema["linked_schema"]}, 
                    output_data=sql)
        return sql
        
    def _format_schema(self, schema: Dict) -> str:
        """格式化schema信息"""
        result = []
        for table in schema["tables"]:
            result.append(f"表名: {table['name']}")
            result.append(f"列: {', '.join(table['columns'])}")
            result.append("")
        return "\n".join(result) 