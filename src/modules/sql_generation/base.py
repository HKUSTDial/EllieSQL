from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from ..base import ModuleBase

class SQLGeneratorBase(ModuleBase):
    def __init__(self, name: str = "SQLGenerator"):
        super().__init__(name)
    """SQL生成模块的基类"""
    
    @abstractmethod
    async def generate_sql(self,
                        query: str,
                        linked_schema: Dict,
                        module_name: Optional[str] = None) -> str:
        """
        根据自然语言查询和链接后的schema生成SQL
        
        Args:
            query: 用户的自然语言查询
            linked_schema: 链接后的schema信息
            
        Returns:
            str: 生成的SQL语句
        """
        pass 
    
    async def generate_sql_with_retry(self, 
                                   query: str, 
                                   schema_linking_output: Dict,
                                   query_id: str = None) -> Tuple[str, str]:
        """使用重试机制生成SQL"""
        return await self.execute_with_retry(
            func=self.generate_sql,
            extract_method='sql',
            error_msg="SQL生成失败",
            query=query,
            schema_linking_output=schema_linking_output,
            query_id=query_id
        ) 