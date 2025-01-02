from ..base import ModuleBase
from typing import Any, Dict, Optional, Tuple, Callable

class PostProcessorBase(ModuleBase):
    """后处理模块的基类"""
    
    async def process_sql_with_retry(self, 
                                  sql: str,
                                  query_id: str = None) -> Tuple[str, str]:
        """使用重试机制处理SQL"""
        return await self.execute_with_retry(
            func=self.process_sql,
            extract_method='sql',
            error_msg="SQL后处理失败",
            sql=sql,
            query_id=query_id
        ) 