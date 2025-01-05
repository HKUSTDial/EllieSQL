from typing import Dict, Optional
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
    
    def clear_cache(self):
        """清除缓存"""
        with self._lock:
            self._schema_cache.clear() 