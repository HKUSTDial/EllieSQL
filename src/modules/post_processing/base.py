from typing import Dict
from ..base import ModuleBase

class PostProcessorBase(ModuleBase):
    """后处理模块的基类"""
    
    def __init__(self, name: str, max_retries: int = 3):
        super().__init__(name)
        self.max_retries = max_retries  # 重试次数阈值
        
    async def process_sql_with_retry(self, sql: str, query_id: str = None) -> str:
        """
        带重试机制的SQL后处理，达到重试阈值后返回原始SQL
        
        Args:
            sql: 待处理的SQL
            query_id: 查询ID
            
        Returns:
            str: 处理后的SQL或原始SQL
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw_processed_sql = await self.process_sql(sql, query_id)
                if raw_processed_sql is not None:
                    processed_sql = self.extractor.extract_sql(raw_processed_sql)
                    if processed_sql is not None:
                        return processed_sql
            except Exception as e:
                last_error = e
                self.logger.warning(f"SQL后处理第{attempt + 1}次尝试失败: {str(e)}")
                continue
        
        # 达到重试阈值，返回原始SQL，并记录错误
        self.logger.error(f"SQL后处理 共{self.max_retries}次尝试后失败，使用原始SQL。最后一次错误: {str(last_error)}。Question ID: {query_id} 程序继续执行...")
        
        # 加载SQL生成的结果以获取原始查询
        try:
            prev_result = self.load_previous_result(query_id)
            original_query = prev_result["input"]["query"]
        except:
            original_query = "Unknown"
            
        # 保存中间结果
        self.save_intermediate(
            input_data={
                "original_query": original_query,
                "original_sql": sql
            },
            output_data={
                "raw_output": "Post-processing failed, using original SQL",
                "processed_sql": sql  # 使用原始SQL
            },
            model_info={
                "model": "none",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            query_id=query_id
        )
        
        return sql 