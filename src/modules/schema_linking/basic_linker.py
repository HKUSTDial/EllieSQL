from typing import Dict
from ...core.llm import LLMBase
from .base import SchemaLinkerBase
from .prompts.schema_prompts import BASE_SCHEMA_SYSTEM, BASE_SCHEMA_USER
from ...core.schema.manager import SchemaManager

class BasicSchemaLinker(SchemaLinkerBase):
    """基础的Schema Linking实现"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 1000,
                max_retries: int = 3):
        super().__init__("BasicSchemaLinker", max_retries)
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.schema_manager = SchemaManager()
        
    async def link_schema(self, query: str, database_schema: Dict, query_id: str = None) -> str:
        """执行schema linking"""
        schema_str = self._format_basic_schema(database_schema)
        
        # 构建prompt
        messages = [
            {"role": "system", "content": BASE_SCHEMA_SYSTEM},
            {"role": "user", "content": BASE_SCHEMA_USER.format(
                schema_str=schema_str,
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
        
        # 提取结果并格式化
        raw_output = result["response"]
        extracted_linked_schema = self.extractor.extract_schema_json(raw_output)
        formatted_linked_schema = self.schema_manager.format_linked_schema(extracted_linked_schema)
        
        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query, 
                # "database_schema": database_schema
            },
            output_data={
                "raw_output": raw_output,
                "extracted_linked_schema": extracted_linked_schema,
                "formatted_linked_schema": formatted_linked_schema
            },
            model_info={
                "model": self.model,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"]
            },
            query_id=query_id
        )
        
        self.log_io(
            {
                "query": query, 
                "database_schema": database_schema, 
                "messages": messages
            }, 
            raw_output
        )
        return raw_output 