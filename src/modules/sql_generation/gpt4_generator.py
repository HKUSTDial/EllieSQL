from typing import Dict, List
from .base import SQLGeneratorBase
from ...core.llm import LLMBase

class GPT4SQLGenerator(SQLGeneratorBase):
    """使用GPT-4生成SQL的实现"""
    
    def __init__(self, llm: LLMBase):
        self.llm = llm
        
    async def generate_sql(self, query: str, linked_schema: Dict) -> str:
        """
        使用GPT-4生成SQL
        
        Args:
            query: 用户查询
            linked_schema: Schema Linking的结果
        """
        # 构建提示词
        schema_str = self._format_schema(linked_schema["original_schema"])
        messages = [
            {"role": "system", "content": """你是一个SQL专家，根据用户的自然语言查询生成准确的SQL语句。
遵循以下规则：
1. 只返回SQL语句，不要有任何解释
2. 使用标准SQL语法
3. 确保SQL语句的表名和列名与schema完全匹配
4. 使用适当的JOIN操作连接表
"""},
            {"role": "user", "content": f"""
数据库Schema:
{schema_str}

相关的表和列:
表: {', '.join(linked_schema['linked_schema']['tables'])}
列: {', '.join(linked_schema['linked_schema']['columns'])}

用户查询: {query}

生成SQL:"""}
        ]
        
        # 调用LLM
        result = await self.llm.call_llm(messages, "gpt4")
        return result["response"].strip()
        
    def _format_schema(self, schema: Dict) -> str:
        """格式化schema信息"""
        result = []
        for table in schema["tables"]:
            result.append(f"表名: {table['name']}")
            result.append(f"列: {', '.join(table['columns'])}")
            result.append("")
        return "\n".join(result) 