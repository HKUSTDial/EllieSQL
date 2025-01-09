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
            result.append(f"数据库: {schema_dict['database']}\n")
            
            # 格式化每个表的信息
            for table in schema_dict['tables']:
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
            if schema_dict.get('foreign_keys'):
                result.append("外键关系:")
                for fk in schema_dict['foreign_keys']:
                    result.append(
                        f"  {fk['table'][0]}.{fk['column'][0]} = "
                        f"{fk['table'][1]}.{fk['column'][1]}"
                    )
                    
            return "\n".join(result)
            
        except Exception as e:
            raise ValueError(f"获取schema失败: {str(e)}")
    
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
            raise ValueError(f"获取schema失败: {str(e)}")
    
    def clear_cache(self):
        """清除缓存"""
        with self._lock:
            self._schema_cache.clear() 