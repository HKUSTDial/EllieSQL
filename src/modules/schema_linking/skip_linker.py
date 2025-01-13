from typing import Dict
from .base import SchemaLinkerBase
from ...core.schema.manager import SchemaManager

class SkipSchemaLinker(SchemaLinkerBase):
    """跳过Schema Linking的实现，直接返回 *带有补充信息* 的数据库schema"""
    
    def __init__(self):
        super().__init__("SkipSchemaLinker")
        self.schema_manager = SchemaManager()
        
    async def link_schema(self, query: str, database_schema: Dict, query_id: str = None) -> str:
        """
        跳过schema linking，直接返回带有补充信息的数据库schema
        
        Args:
            query: 用户查询
            database_schema: 数据库schema
            query_id: 查询ID
            
        Returns:
            str: 格式化的schema字符串
        """
        try:
            # 构造一个包含所有表和列的linked_schema
            linked_schema = {
                "tables": [
                    {
                        "table": table["table"],
                        "columns": list(table["columns"].keys()),
                        "columns_info": table["columns"]
                    }
                    for table in database_schema["tables"]
                ]
            }
            
            # 格式化linked schema
            formatted_linked_schema = self.schema_manager.format_linked_schema(linked_schema)
            # print("****formatted_linked_schema*****\n", formatted_linked_schema, "\n*********")
            # 保存中间结果
            self.save_intermediate(
                input_data={
                    "query": query,
                    # "database_schema": database_schema
                },
                output_data={
                    # "linked_schema": linked_schema,
                    "formatted_linked_schema": formatted_linked_schema
                },
                model_info={
                    "model": "none",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                },
                query_id=query_id
            )

            # 为保持接口一致性，使用用于占位的固定raw_output内容
            raw_output = """```json {"tables": [{"table": "<skip schema linking>", "columns": ["<skip schema linking>"]}]}```"""

            self.log_io(
                {
                    "query": query, 
                    "database_schema": database_schema, 
                    "formatted_schema": formatted_linked_schema
                }, 
                raw_output
            )
            return raw_output
            
        except Exception as e:
            self.logger.error(f"Skip schema linking failed: {str(e)}")
            raise 