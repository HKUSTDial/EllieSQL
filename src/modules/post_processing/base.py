from typing import Dict
from ..base import ModuleBase

class PostProcessorBase(ModuleBase):
    """后处理模块的基类"""
    
    def __init__(self, name: str, max_retries: int = 3):
        super().__init__(name)
        self.max_retries = max_retries  # 重试次数阈值
        
    async def process_sql_with_retry(self, sql: str, query_id: str = None) -> str:
        """
        带重试机制的SQL后处理
        
        Args:
            sql: 待处理的SQL
            query_id: 查询ID
            
        Returns:
            str: 处理后的SQL
            
        Raises:
            ValueError: 当重试次数达到上限时抛出
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = await self.process_sql(sql, query_id)
                if result is not None:
                    extracted = self.extractor.extract_sql(result)
                    if extracted is not None:
                        return result
            except Exception as e:
                last_error = e
                self.logger.warning(f"SQL后处理第{attempt + 1}次尝试失败: {str(e)}")
                continue
                
        error_message = f"SQL后处理在{self.max_retries}次尝试后失败。最后一次错误: {str(last_error)}"
        self.logger.error(error_message)
        raise ValueError(error_message) 