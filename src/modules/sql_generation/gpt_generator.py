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
        
    def _format_table_info(self, table: Dict) -> str:
        """格式化单个表的信息"""
        info = [f"表 '{table['table']}'："]
        info.append(f"  相关列: {', '.join(table['columns'])}")
        
        if "primary_keys" in table:
            info.append(f"  主键: {', '.join(table['primary_keys'])}")
            
        if "foreign_keys" in table and table["foreign_keys"]:
            info.append("  外键关系:")
            for fk in table["foreign_keys"]:
                info.append(f"    - {table['table']}.{fk['column']} -> "
                          f"{fk['referenced_table']}.{fk['referenced_column']}")
                
        return "\n".join(info)
    
    def _format_relations(self, linked_tables: List[Dict]) -> str:
        """格式化表之间的关系信息"""
        relations = []
        for table in linked_tables:
            if "foreign_keys" in table and table["foreign_keys"]:
                for fk in table["foreign_keys"]:
                    relations.append(
                        f"- {table['table']}.{fk['column']} = "
                        f"{fk['referenced_table']}.{fk['referenced_column']}"
                    )
        return "\n".join(relations) if relations else "无需表连接"
        
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
        linked_tables = schema_linking_output["linked_schema"]["tables"]
        
        # 构建提示词
        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": SQL_GENERATION_USER.format(
                tables="\n".join([
                    self._format_table_info(t) for t in linked_tables
                ]),
                relations=self._format_relations(linked_tables),
                query=query
            )}
        ]
        
        # print('\n'+messages[0]['content']+'\n')
        # print('\n'+messages[1]['content']+'\n')
        result = await self.llm.call_llm(
            messages,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        raw_output = result["response"]
        extracted_sql = self.extractor.extract_sql(raw_output)
        
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
        