from abc import ABC, abstractmethod
from typing import Dict, List

class PostProcessorBase(ABC):
    """后处理模块的基类"""
    
    @abstractmethod
    async def process_sql(self, sql: str) -> str:
        """
        对生成的SQL进行后处理
        
        Args:
            sql: 待处理的SQL语句
            
        Returns:
            str: 处理后的SQL语句
        """
        pass 