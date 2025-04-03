from typing import Dict
from ..base import ModuleBase

class PostProcessorBase(ModuleBase):
    """Base class for post-processing modules"""
    
    def __init__(self, name: str, max_retries: int = 3):
        super().__init__(name)
        self.max_retries = max_retries  # Retry threshold
        
    async def process_sql_with_retry(self, sql: str, query_id: str = None) -> str:
        """
        SQL post-processing with retry mechanism, returning the original SQL after reaching the retry threshold
        
        :param sql: The SQL to be processed
        :param query_id: The query ID
            
        :return: The processed SQL or the original SQL
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
                self.logger.warning(f"SQL post-processing failed on the {attempt + 1}/{self.max_retries} attempt: {str(e)}")
                continue
        
        # Reached the retry threshold, return the original SQL, and record the error
        self.logger.error(f"SQL post-processing failed after {self.max_retries} attempts, using the original SQL. Last error: {str(last_error)}. Question ID: {query_id} The program continues to execute...")
        
        # Load the SQL generated result to get the original query
        try:
            prev_result = self.load_previous_result(query_id)
            original_query = prev_result["input"]["query"]
        except:
            original_query = "Unknown"
            
        # Save the intermediate result
        self.save_intermediate(
            input_data={
                "original_query": original_query,
                "original_sql": sql
            },
            output_data={
                "raw_output": "Post-processing failed, using original SQL",
                "processed_sql": sql  # Use the original SQL
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