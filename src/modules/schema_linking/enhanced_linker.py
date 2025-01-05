from typing import Dict
from .base import SchemaLinkerBase
from ...core.llm import LLMBase
from .prompts.schema_prompts import ENHANCED_SCHEMA_SYSTEM, BASE_SCHEMA_USER

class EnhancedSchemaLinker(SchemaLinkerBase):
    """使用增强schema信息的Schema Linker"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 1000):
        super().__init__("EnhancedSchemaLinker")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def _format_enhanced_schema(self, schema: Dict) -> str:
        """格式化增强的schema信息"""
        result = []
        
        # 添加数据库名称
        result.append(f"数据库: {schema['database']}\n")
        
        # 格式化每个表的信息
        for table in schema['tables']:
            result.append(f"表名: {table['table']}")
            
            # 添加列信息
            result.append("列:")
            for col_name, col_info in table['columns'].items():
                col_desc = []
                if col_info['expanded_name']:
                    col_desc.append(f"含义: {col_info['expanded_name']}")
                if col_info['description']:
                    col_desc.append(f"描述: {col_info['description']}")
                if col_info['data_format']:
                    col_desc.append(f"格式: {col_info['data_format']}")
                if col_info['value_description']:
                    col_desc.append(f"值描述: {col_info['value_description']}")
                if col_info['value_examples']:
                    col_desc.append(f"示例值: {', '.join(col_info['value_examples'])}")
                    
                if col_desc:
                    result.append(f"  - {col_name} ({col_info['type']}): {' | '.join(col_desc)}")
                else:
                    result.append(f"  - {col_name} ({col_info['type']})")
                    
            # 添加主键信息
            if table['primary_keys']:
                result.append(f"主键: {', '.join(table['primary_keys'])}")
                
            result.append("")  # 添加空行分隔
            
        # 添加外键关系
        if schema.get('foreign_keys'):
            result.append("外键关系:")
            for fk in schema['foreign_keys']:
                result.append(
                    f"  {fk['table'][0]}.{fk['column'][0]} = "
                    f"{fk['table'][1]}.{fk['column'][1]}"
                )
                
        return "\n".join(result)
        
    async def link_schema(self, query: str, database_schema: Dict, query_id: str = None) -> str:
        """执行schema linking"""
        schema_str = self._format_enhanced_schema(database_schema)
        
        # 记录格式化后的schema以便检查
        self.logger.debug("格式化后的Schema信息:")
        self.logger.debug("\n" + schema_str)
        
        messages = [
            {"role": "system", "content": ENHANCED_SCHEMA_SYSTEM},
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
            input_data={
                "query": query, 
                "database_schema": database_schema,
                "formatted_schema": schema_str,  # 保存格式化后的schema
                "messages": messages
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
        
        self.log_io(
            {
                "query": query, 
                "schema": database_schema, 
                "formatted_schema": schema_str,
                "messages": messages
            }, 
            raw_output
        )
        
        return raw_output 