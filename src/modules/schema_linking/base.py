from abc import ABC, abstractmethod
from typing import Dict, List
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