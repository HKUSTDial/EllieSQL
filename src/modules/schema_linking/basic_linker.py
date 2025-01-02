from typing import Dict, List, Optional
from .base import SchemaLinkerBase
from ...core.llm import LLMBase
from .prompts.basic_prompts import SCHEMA_LINKING_SYSTEM, SCHEMA_LINKING_USER

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
        
    async def link_schema(self, query: str, database_schema: Dict, query_id: str = None) -> str:
        """
        基于简单提示词的schema linking
        
        Args:
            query: 用户查询
            database_schema: 数据库schema
            query_id: 查询ID
            
        Returns:
            str: 原始输出，包含JSON格式的schema linking结果
        """
        schema_str = self._format_schema(database_schema)
        messages = [
            {"role": "system", "content": SCHEMA_LINKING_SYSTEM},
            {"role": "user", "content": SCHEMA_LINKING_USER.format(
                schema_str=schema_str,
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
        extracted_schema = self.extractor.extract_schema_json(raw_output)
        
        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query, 
                "database_schema": database_schema
            },
            output_data={
                "raw_output": raw_output,
                "extracted_schema": extracted_schema
            },
            model_info={
                "model": self.model,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"]
            },
            query_id=query_id
        )
        
        self.log_io({"query": query, "schema": database_schema, "messages": messages}, raw_output)
        return raw_output
        
    def _format_schema(self, schema: Dict) -> str:
        """格式化schema信息，清晰展示表和列的归属关系"""
        result = []
        for table in schema["tables"]:
            result.append(f"表名: {table['name']}")
            result.append(f"该表的列: {', '.join(table['columns'])}")
            result.append("")
        return "\n".join(result) 