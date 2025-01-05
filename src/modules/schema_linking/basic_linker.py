from typing import Dict
from .base import SchemaLinkerBase
from ...core.llm import LLMBase
from .prompts.schema_prompts import BASE_SCHEMA_SYSTEM, BASE_SCHEMA_USER

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
        """执行schema linking"""
        schema_str = self._format_basic_schema(database_schema)
        
        messages = [
            {"role": "system", "content": BASE_SCHEMA_SYSTEM},
            {"role": "user", "content": BASE_SCHEMA_USER.format(
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
            input_data={"query": query, "database_schema": database_schema},
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