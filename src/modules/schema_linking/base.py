from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from ..base import ModuleBase

class SchemaLinkerBase(ModuleBase):
    """Schema Linking模块的基类"""
    
    def __init__(self, name: str = "SchemaLinker"):
        super().__init__(name)
    
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
        return await self.execute_with_retry(
            func=self.link_schema,
            extract_method='schema_json',
            error_msg="Schema linking失败",
            query=query,
            database_schema=database_schema,
            query_id=query_id
        ) 