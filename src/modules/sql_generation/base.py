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
    
    async def generate_sql_with_retry(self, query: str, schema_linking_output: Dict, query_id: str, module_name: str = None) -> str:
        """
        带重试机制的SQL生成
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
            query_id: 查询ID
            module_name: 模块名称
            
        Returns:
            str: 生成的SQL
            
        Raises:
            ValueError: 当重试次数达到上限时抛出
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = await self.generate_sql(query, schema_linking_output, query_id, module_name)
                if result is not None:
                    extracted = self.extractor.extract_sql(result)
                    if extracted is not None:
                        return result
            except Exception as e:
                last_error = e
                self.logger.warning(f"SQL生成第{attempt + 1}次尝试失败: {str(e)}")
                continue
                
        error_message = f"SQL生成在{self.max_retries}次尝试后失败。最后一次错误: {str(last_error)}"
        self.logger.error(error_message)
        raise ValueError(error_message) 