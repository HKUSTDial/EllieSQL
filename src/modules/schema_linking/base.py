from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from ..base import ModuleBase
import json

class SchemaLinkerBase(ModuleBase):
    """Schema Linking模块的基类"""
    
    def __init__(self, name: str = "SchemaLinker"):
        super().__init__(name)
    
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
    
    async def link_schema_with_retry(self, 
                                  query: str, 
                                  database_schema: Dict,
                                  query_id: str = None) -> Tuple[str, Dict]:
        """使用重试机制进行schema linking"""
        raw_output, extracted_schema = await self.execute_with_retry(
            func=self.link_schema,
            extract_method='schema_json',
            error_msg="Schema linking失败",
            query=query,
            database_schema=database_schema,
            query_id=query_id
        )
        
        # 补充主键和外键信息
        enriched_schema = self.enrich_schema_info(extracted_schema, database_schema)
        
        # 输出enrich后的schema以便检查
        print("\n" + "="*50)
        print("Enriched Schema:")
        print(json.dumps(enriched_schema, indent=2, ensure_ascii=False))
        print("="*50 + "\n")
        
        return raw_output, enriched_schema 