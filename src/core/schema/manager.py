from typing import Dict, Optional, Union
from pathlib import Path
from threading import Lock
from .schema import DatabaseSchema
from .extractor import SchemaExtractor

class SchemaManager:
    """管理数据库schema的单例类"""
    _instance = None
    _lock = Lock()
    _schema_cache: Dict[str, DatabaseSchema] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_schema(self, db_id: str, db_path: str) -> DatabaseSchema:
        """
        获取数据库schema，优先从缓存获取
        
        Args:
            db_id: 数据库ID
            db_path: 数据库文件路径
            
        Returns:
            DatabaseSchema: 数据库schema
        """
        if db_id not in self._schema_cache:
            with self._lock:
                if db_id not in self._schema_cache:
                    self._schema_cache[db_id] = SchemaExtractor.extract_from_db(db_path, db_id)
        return self._schema_cache[db_id]
    
    def get_formatted_enriched_schema(self, db_id: str, source: str) -> str:
        """
        根据数据库ID和来源获取格式化的schema字符串
        
        Args:
            db_id: 数据库ID
            source: 数据库来源
            
        Returns:
            str: 格式化的schema字符串
        """
        # 构建数据库路径
        db_folder = f"{source}_{db_id}"
        db_file = f"{db_id}.sqlite"
        db_path = f"./data/merged_databases/{db_folder}/{db_file}"
        
        try:
            schema = self.get_schema(db_id, db_path)
            schema_dict = schema.to_dict()
            
            # 格式化schema信息
            result = []
            
            # 添加数据库名称
            result.append(f"Database: {schema_dict['database']}\n")
            
            # 格式化每个表的信息
            for table in schema_dict['tables']:
                result.append(f"Table name: {table['table']}")
                
                # 添加列信息
                result.append("Columns:")
                for col_name, col_info in table['columns'].items():
                    col_desc = []
                    if col_info['expanded_name']:
                        col_desc.append(f"Meaning: {col_info['expanded_name']}")
                    if col_info['description']:
                        col_desc.append(f"Description: {col_info['description']}")
                    if col_info['data_format']:
                        col_desc.append(f"Format: {col_info['data_format']}")
                    if col_info['value_description']:
                        col_desc.append(f"Value description: {col_info['value_description']}")
                    if col_info['value_examples']:
                        col_desc.append(f"Value examples: {', '.join(col_info['value_examples'])}")
                        
                    if col_desc:
                        result.append(f"  - {col_name} ({col_info['type']}): {' | '.join(col_desc)}")
                    else:
                        result.append(f"  - {col_name} ({col_info['type']})")
                        
                # 添加主键信息
                if table['primary_keys']:
                    result.append(f"Primary key: {', '.join(table['primary_keys'])}")
                    
                result.append("")  # 添加空行分隔
                
            # 添加外键关系
            if schema_dict.get('foreign_keys'):
                result.append("Foreign keys:")
                for fk in schema_dict['foreign_keys']:
                    result.append(
                        f"  {fk['table'][0]}.{fk['column'][0]} = "
                        f"{fk['table'][1]}.{fk['column'][1]}"
                    )
                    
            return "\n".join(result)
            
        except Exception as e:
            raise ValueError(f"Access schema failure: {str(e)}")
    
    def get_db_schema_by_id_source(self, db_id: str, source: str) -> Dict:
        """
        根据数据库ID和来源获取包含补充信息的schema
        
        Args:
            db_id: 数据库ID
            source: 数据库来源
            
        Returns:
            Dict: 包含补充信息的schema字典
        """
        # 构建数据库路径
        db_folder = f"{source}_{db_id}"  # 例如: spider_dev_academic
        db_file = f"{db_id}.sqlite"      # 例如: academic.sqlite
        db_path = f"./data/merged_databases/{db_folder}/{db_file}"
        
        try:
            # 获取schema并转换为字典格式
            schema = self.get_schema(db_id, db_path)
            return schema.to_dict()
        except Exception as e:
            raise ValueError(f"Access schema failure: {str(e)}")
    
    def clear_cache(self):
        """清除缓存"""
        with self._lock:
            self._schema_cache.clear() 
    
    def format_enriched_db_schema(self, schema: Dict) -> str:
        """
        格式化增强的数据库schema信息，用于SQL生成模块的提示词
        
        Args:
            schema: 数据库schema字典
            
        Returns:
            str: 格式化后的schema字符串
        """
        result = []
        
        # 添加数据库名称
        result.append(f"Database: {schema['database']}\n")
        
        # 格式化每个表的信息
        for table in schema['tables']:
            result.append(f"Table name: {table['table']}")
            
            # 添加列信息
            result.append("Columns:")
            for col_name, col_info in table['columns'].items():
                col_desc = []
                if col_info['expanded_name']:
                    col_desc.append(f"Meaning: {col_info['expanded_name']}")
                if col_info['description']:
                    col_desc.append(f"Description: {col_info['description']}")
                if col_info['data_format']:
                    col_desc.append(f"Format: {col_info['data_format']}")
                if col_info['value_description']:
                    col_desc.append(f"Value description: {col_info['value_description']}")
                if col_info['value_examples']:
                    col_desc.append(f"Value examples: {', '.join(col_info['value_examples'])}")
                    
                if col_desc:
                    result.append(f"  - {col_name} ({col_info['type']}): {' | '.join(col_desc)}")
                else:
                    result.append(f"  - {col_name} ({col_info['type']})")
                    
            # 添加主键信息
            if table['primary_keys']:
                result.append(f"Primary key: {', '.join(table['primary_keys'])}")
                
            result.append("")  # 添加空行分隔
            
        # 添加外键关系
        if schema.get('foreign_keys'):
            result.append("Foreign keys:")
            for fk in schema['foreign_keys']:
                result.append(
                    f"  {fk['table'][0]}.{fk['column'][0]} = "
                    f"{fk['table'][1]}.{fk['column'][1]}"
                )
                
        return "\n".join(result)
    
    def format_linked_schema(self, linked_schema: Dict) -> str:
        """
        格式化linked schema为SQL生成模块使用的格式，用于SQL生成模块的提示词
        
        Args:
            linked_schema: Schema Linking后的schema
            
        Returns:
            str: 格式化后的schema字符串
        """
        result = []
        
        # 格式化每个表的信息
        for table in linked_schema["tables"]:
            table_name = table["table"]
            columns = table.get("columns", [])
            columns_info = table.get("columns_info", {})
            
            # 添加表信息
            result.append(f"Table: {table_name}")
            
            # 添加列信息（包含补充信息）
            if columns:
                result.append("Columns:")
                for col in columns:
                    col_info = columns_info.get(col, {})
                    col_desc = []
                    
                    # 添加列的类型
                    col_type = col_info.get("type", "")
                    if col_type:
                        col_desc.append(f"Type: {col_type}")
                    
                    # 添加其他补充信息
                    if col_info.get("expanded_name"):
                        col_desc.append(f"Meaning: {col_info['expanded_name']}")
                    if col_info.get("description"):
                        col_desc.append(f"Description: {col_info['description']}")
                    if col_info.get("data_format"):
                        col_desc.append(f"Format: {col_info['data_format']}")
                    if col_info.get("value_description"):
                        col_desc.append(f"Value description: {col_info['value_description']}")
                    if col_info.get("value_examples"):
                        col_desc.append(f"Value examples: {', '.join(col_info['value_examples'])}")
                    
                    if col_desc:
                        result.append(f"  - {col}: {' | '.join(col_desc)}")
                    else:
                        result.append(f"  - {col}")
            
            # 添加主键信息
            if "primary_keys" in table:
                result.append(f"主键: {', '.join(table['primary_keys'])}")
            
            # 添加外键信息
            if "foreign_keys" in table:
                result.append("Foreign keys:")
                for fk in table["foreign_keys"]:
                    result.append(
                        f"  - {fk['column']} -> "
                        f"{fk['referenced_table']}.{fk['referenced_column']}"
                    )
            
            result.append("")  # 添加空行分隔
            
        return "\n".join(result) 