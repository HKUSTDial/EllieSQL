from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from ..base import ModuleBase
import json
import os

class SchemaLinkerBase(ModuleBase):
    """Schema Linking模块的基类"""
    
    def __init__(self, name: str, max_retries: int = 3):
        super().__init__(name)
        self.max_retries = max_retries  # 重试次数阈值
        
    def enrich_schema_info(self, linked_schema: Dict, database_schema: Dict) -> Dict:
        """
        为linked schema补充主键和外键信息
        
        Args:
            linked_schema: Schema Linking的原始输出
            database_schema: 完整的数据库schema
            
        Returns:
            Dict: 补充了主键和外键信息的schema
        """
        # 创建表名到表信息的映射
        db_tables = {table["table"]: table for table in database_schema["tables"]}
        
        # 为每个linked table补充信息
        for table in linked_schema["tables"]:
            table_name = table["table"]
            if table_name in db_tables:
                # 补充主键信息
                if "primary_keys" in db_tables[table_name]:
                    table["primary_keys"] = db_tables[table_name]["primary_keys"]
                
        # 补充外键信息
        if "foreign_keys" in database_schema:
            for fk in database_schema["foreign_keys"]:
                # 检查外键相关的表是否都在linked schema中
                src_table = fk["table"][0]
                dst_table = fk["table"][1]
                linked_tables = [t["table"] for t in linked_schema["tables"]]
                
                if src_table in linked_tables and dst_table in linked_tables:
                    # 找到源表
                    for table in linked_schema["tables"]:
                        if table["table"] == src_table:
                            if "foreign_keys" not in table:
                                table["foreign_keys"] = []
                            # 添加外键信息
                            table["foreign_keys"].append({
                                "column": fk["column"][0],
                                "referenced_table": dst_table,
                                "referenced_column": fk["column"][1]
                            })
                            
        return linked_schema
    
    @abstractmethod
    async def link_schema(self, 
                       query: str, 
                       database_schema: Dict) -> Dict:
        """
        将自然语言查询与数据库schema进行链接
        
        Args:
            query: 用户的自然语言查询
            database_schema: 数据库schema信息
            
        Returns:
            Dict: 链接后的schema信息
        """
        pass
    
    async def link_schema_with_retry(self, query: str, database_schema: Dict, query_id: str = None) -> str:
        """
        带重试机制的schema linking，达到重试阈值后返回完整的数据库schema
        
        Args:
            query: 用户查询
            database_schema: 数据库schema
            query_id: 查询ID
            
        Returns:
            str: Schema Linking的结果或完整数据库schema的JSON字符串
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw_schema_output = await self.link_schema(query, database_schema, query_id)
                if raw_schema_output is not None:
                    # 从原始输出中提取schema并补充主键和外键信息
                    extracted_linked_schema = self.extractor.extract_schema_json(raw_schema_output)
                    enriched_linked_schema = self.enrich_schema_info(extracted_linked_schema, database_schema)
                    if extracted_linked_schema is not None:
                        return enriched_linked_schema
            except Exception as e:
                last_error = e
                self.logger.warning(f"Schema linking 第{attempt + 1}/{self.max_retries}次尝试失败: {str(e)}")
                continue
        
        # 达到重试阈值，返回完整的数据库schema，并确保格式与正常返回一致
        self.logger.error(f"Schema linking 共{self.max_retries}次尝试后失败，返回完整数据库schema。最后一次错误: {str(last_error)}。Question ID: {query_id} 程序继续执行...")
        
        # 构造一个包含所有表和列的schema，并补充主键和外键信息
        full_schema = {
            "tables": [
                {
                    "table": table["table"],
                    "columns": list(table["columns"].keys()),
                    "columns_info": table["columns"],
                    "primary_keys": table.get("primary_keys", [])
                }
                for table in database_schema["tables"]
            ]
        }
        
        # 添加外键信息
        if "foreign_keys" in database_schema:
            for table in full_schema["tables"]:
                table_name = table["table"]
                table["foreign_keys"] = []
                for fk in database_schema["foreign_keys"]:
                    if fk["table"][0] == table_name:
                        table["foreign_keys"].append({
                            "column": fk["column"][0],
                            "referenced_table": fk["table"][1],
                            "referenced_column": fk["column"][1]
                        })
        
        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query,
                # "database_schema": database_schema
            },
            output_data={
                "raw_output": "Schema linking failed, using full database schema",
                "extracted_linked_schema": full_schema,
                "formatted_linked_schema": self.schema_manager.format_linked_schema(full_schema)
            },
            model_info={
                "model": "none",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            query_id=query_id
        )
        # print(self.schema_manager.format_linked_schema(full_schema))
        return full_schema
    
    def _format_basic_schema(self, schema: Dict) -> str:
        """基础schema格式化方法"""
        result = []
        
        # 添加数据库名称
        result.append(f"Database: {schema['database']}\n")
        
        # 格式化表结构
        for table in schema['tables']:
            result.append(f"Table name: {table['table']}")
            result.append(f"Columns: {', '.join(table['columns'])}")
            if table['primary_keys']:
                result.append(f"Primary keys: {', '.join(table['primary_keys'])}")
            result.append("")

        # 格式化外键关系
        if schema.get('foreign_keys'):
            result.append("Foreign keys:")
            for fk in schema['foreign_keys']:
                result.append(
                    f"  {fk['table'][0]}.{fk['column'][0]} = "
                    f"{fk['table'][1]}.{fk['column'][1]}"
                )
                
        return "\n".join(result) 
    
    def save_linked_schema_result(self, query_id: str, source: str, linked_schema: Dict) -> None:
        """
        Save the linked schema result to a separate JSONL file.
        
        Args:
            query_id: Query ID
            source: Data source (e.g., 'bird_dev')
            linked_schema: The linked schema with primary/foreign keys
        """
        # Get the pipeline directory from the intermediate result handler
        pipeline_dir = self.intermediate.pipeline_dir
        
        # Define the output file path
        output_file = os.path.join(pipeline_dir, "linked_schema_results.jsonl")
        
        # Format the result in the specified structure
        result = {
            "query_id": query_id,
            "source": source,
            "database": linked_schema.get("database", ""),
            "tables": [
                {
                    "table": table["table"],
                    "columns": table["columns"]
                }
                for table in linked_schema.get("tables", [])
            ]
        }
        
        # Save to JSONL file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n") 