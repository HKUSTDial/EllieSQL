from typing import Dict, List
from .base import SQLGeneratorBase
from ...core.llm import LLMBase
from .prompts.gpt_prompts import SQL_GENERATION_SYSTEM, SQL_GENERATION_USER
import re

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
        
    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """
        使用GPT模型生成SQL

        Original Schema格式：
        {
            "tables": [{"name": "students", "columns": ["id", "name", "age", "class_id"]}, {"name": "classes", "columns": ["id", "name", "teacher_id"]}]
        }
        
        Linked Schema格式：
        {
            "tables": ["students", "classes"],
            "columns": ["age", "name", "class_id", "name"]
        }
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking模块的输出，包含原始schema和linked schema
            query_id: 查询ID
        """
        # 加载schema linking的结果
        if not schema_linking_output:
            prev_result = self.load_previous(query_id, "BasicSchemaLinker")
            schema_linking_output = prev_result["output"]
        
        # 从linked_schema中提取表和列信息
        tables = schema_linking_output["linked_schema"]["tables"]
        columns = schema_linking_output["linked_schema"]["columns"]
        schema = schema_linking_output["original_schema"]
        
        # 构建完整的schema字符串
        # schema_str = self._format_schema(schema)
        
        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": SQL_GENERATION_USER.format(
                # schema_str=schema_str,
                tables=", ".join(tables),
                columns=", ".join(columns),
                query=query
            )}
        ]
        
        result = await self.llm.call_llm(
            messages,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        raw_output = result["response"]
        extracted_sql = self._extract_sql(raw_output)
        
        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query, 
                "original_schema": schema_linking_output["original_schema"],
                "linked_schema": schema_linking_output["linked_schema"],
                # "tables": tables,
                # "columns": columns
            },
            output_data={
                "raw_output": raw_output,
                "extracted_sql": extracted_sql
            },
            model_info={
                "model": self.model,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"]
            },
            query_id=query_id
        )
        self.log_io(input_data={"query": query, "linked_schema": schema_linking_output, "messages": messages}, output_data=raw_output)

        return raw_output
        
    def _format_schema(self, schema: Dict) -> str:
        """格式化完整的schema信息"""
        result = []
        for table in schema["tables"]:
            result.append(f"表名: {table['name']}")
            result.append(f"列: {', '.join(table['columns'])}")
            result.append("")
        return "\n".join(result) 
        
    def _extract_sql(self, text: str) -> str:
        """从文本中提取SQL代码块"""
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip() 