from typing import Dict, List
from .base import SchemaLinkerBase
from ...core.llm import LLMBase

class BasicSchemaLinker(SchemaLinkerBase):
    """基本的Schema Linking实现"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 1000):
        super().__init__("BasicSchemaLinker")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def link_schema(self, query: str, database_schema: Dict) -> Dict:
        """
        基于简单提示词的schema linking
        
        Args:
            query: 用户查询
            database_schema: 数据库schema，格式为:
                {
                    "tables": [
                        {
                            "name": "table_name",
                            "columns": ["col1", "col2", ...]
                        },
                        ...
                    ]
                }
        """
        # 构建提示词
        schema_str = self._format_schema(database_schema)
        messages = [
            {"role": "system", "content": "你是一个SQL专家，帮助识别自然语言查询中涉及的数据库表和列。"},
            {"role": "user", "content": f"""
数据库Schema如下:
{schema_str}

用户查询: {query}

请识别查询中涉及的表和列，以JSON格式返回:
{{
    "tables": ["涉及的表名1", "表名2", ...],
    "columns": ["涉及的列名1", "列名2", ...]
}}
只返回JSON，不要其他解释。
"""}
        ]
        
        # 调用LLM
        result = await self.llm.call_llm(
            messages, 
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        output = {
            "original_schema": database_schema,
            "linked_schema": eval(result["response"])
        }
        
        self.log_io({"query": query, "schema": database_schema}, output)
        return output
        
    def _format_schema(self, schema: Dict) -> str:
        """格式化schema信息"""
        result = []
        for table in schema["tables"]:
            result.append(f"表名: {table['name']}")
            result.append(f"列: {', '.join(table['columns'])}")
            result.append("")
        return "\n".join(result) 