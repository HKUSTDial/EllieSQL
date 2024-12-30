from abc import ABC, abstractmethod
from typing import Dict, List

class SQLGeneratorBase(ABC):
    """SQL生成模块的基类"""
    
    @abstractmethod
    async def generate_sql(self,
                        query: str,
                        linked_schema: Dict) -> str:
        """
        根据自然语言查询和链接后的schema生成SQL
        
        Args:
            query: 用户的自然语言查询
            linked_schema: 链接后的schema信息
            
        Returns:
            str: 生成的SQL语句
        """
        pass 