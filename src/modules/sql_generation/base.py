from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from ..base import ModuleBase
from ...core.sql_execute import validate_sql_execution
from ...core.utils import load_json
from ...core.config import Config
import os

class SQLGeneratorBase(ModuleBase):
    """SQL生成模块的基类"""
    
    def __init__(self, name: str = "SQLGenerator", max_retries: int = 3):
        super().__init__(name)
        self.max_retries = max_retries
        
    @abstractmethod
    async def generate_sql(self,
                        query: str,
                        schema_linking_output: Dict,
                        module_name: Optional[str] = None) -> str:
        """
        Generate SQL, return the raw output of the generated SQL wrapped in a code block, 
        use the retry mechanism to safely extract the SQL from the code block in generate_sql_with_retry
        
        :param query: User query
        :param schema_linking_output: Output of Schema Linking
                                    schema_linking_output = {
                                        "original_schema": database_schema,
                                        "linked_schema": enriched_linked_schema (if enhanced linker is used, the primary and foreign keys are already supplemented, while the basic linker does not have them)
                                    }
        :param module_name: Module name
            
        :return: The raw output of the generated SQL,供generate_sql_with_retry提取, 例如: Therefore, the final SQL is ```sql SELECT * FROM players;```
        """
        pass 
    
    async def generate_sql_with_retry(self, query: str, schema_linking_output: Dict, query_id: str, module_name: str = None) -> str:
        """
        SQL generation with retry mechanism and extraction, only used to prevent the situation where the SQL cannot be extracted, 
        without validating the SQL's executability
        
        :param query: User query
        :param schema_linking_output: Output of Schema Linking
        :param query_id: Query ID
        :param module_name: Module name
            
        :return: The generated SQL (if all attempts fail, return the fallback SQL)
        """
        last_error = None
        last_extracted_sql = None  # Save the last successfully extracted SQL
        
        for attempt in range(self.max_retries):
            try:
                raw_sql_output = await self.generate_sql(query, schema_linking_output, query_id, module_name)
                if raw_sql_output is not None:
                    extracted_sql = self.extractor.extract_sql(raw_sql_output)
                    if extracted_sql is not None:
                        return extracted_sql
                    else:
                        self.logger.warning(f"Can not extract SQL from the output, attempt {attempt + 1}/{self.max_retries}")
                        continue
            except Exception as e:
                last_error = e
                self.logger.warning(f"SQL generation failed on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                continue
                
        # Reach the retry threshold, record the error
        error_message = f"SQL generation failed after {self.max_retries} attempts, unable to extract valid SQL. Last error: {str(last_error)}. Question ID: {query_id} program continues..."
        self.logger.error(error_message)
        
        # Get the first table and its first column as fallback
        first_table = schema_linking_output["original_schema"]["tables"][0]
        table_name = first_table["table"]
        first_column = list(first_table["columns"].keys())[0]
        fallback_sql = f"SELECT {first_column} FROM {table_name} LIMIT 1;"
        
        return fallback_sql
    
    async def generate_sql_with_retry_with_validate(self, query: str, schema_linking_output: Dict, query_id: str, module_name: str = None) -> str:
        """
        NOTE: [deprecated] This function is deprecated, validate_sql should be implemented by the refiner
        SQL generation with retry mechanism and validation, return the last valid SQL when reaching the retry threshold, 
        or return the fallback SQL if completely failed.
        
        :param query: User query
        :param schema_linking_output: Output of Schema Linking
        :param query_id: Query ID
        :param module_name: Module name
            
        :return: The generated SQL (if all attempts fail, return the fallback SQL)
        """
        last_error = None
        last_extracted_sql = None  # Save the last successfully extracted SQL
        
        # Get the database path
        if not self.data_file:
            raise ValueError("Data file path not set. Please call set_data_file() first.")
            
        data = load_json(self.data_file) # data_file: e.g. ./data/merge_dev_demo.json
        db_path = None
        
        for item in data:
            if item.get("question_id") == query_id:
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                # Use the relative path to the data_file
                data_dir = os.path.dirname(self.data_file)
                db_folder = f"{source}_{db_id}"
                db_file = f"{db_id}.sqlite"
                db_path = str(Config().database_dir / db_folder / db_file)
                break
                
        if not db_path:
            raise ValueError(f"Can not find the database path for question ID {query_id}")
            
        for attempt in range(self.max_retries):
            try:
                raw_sql_output = await self.generate_sql(query, schema_linking_output, query_id, module_name)
                if raw_sql_output is not None:
                    extracted_sql = self.extractor.extract_sql(raw_sql_output)
                    if extracted_sql is not None:
                        last_extracted_sql = extracted_sql  # Update the last successfully extracted SQL
                        # Validate whether the SQL can be executed and the result is not empty
                        is_valid, error_msg = validate_sql_execution(db_path, extracted_sql)
                        if is_valid:
                            return extracted_sql
                        else:
                            self.logger.warning(f"SQL validation failed: {error_msg}")
                            continue
            except Exception as e:
                last_error = e
                self.logger.warning(f"SQL generation failed on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                continue
                
        # Reach the retry threshold, record the error and return the last valid SQL
        if last_extracted_sql:
            error_message = f"SQL generation failed after {self.max_retries} attempts, using the last successfully extracted SQL. Last error: {str(last_error)}. Question ID: {query_id} program continues..."
            self.logger.error(error_message)
            return last_extracted_sql
        else:
            # When completely failed, generate a simple fallback SQL
            error_message = f"SQL generation failed after {self.max_retries} attempts, unable to extract valid SQL, using the fallback SQL. Last error: {str(last_error)}. Question ID: {query_id} program continues..."
            self.logger.error(error_message)
            
            # Get the first table and its first column as fallback, construct and validate the fallback SQL
            first_table = schema_linking_output["original_schema"]["tables"][0]
            table_name = first_table["table"]
            first_column = list(first_table["columns"].keys())[0]
            fallback_sql = f"SELECT {first_column} FROM {table_name} LIMIT 1;"
            is_valid, error_msg = validate_sql_execution(db_path, fallback_sql)
            
            if not is_valid:
                self.logger.fatal(f"[FATAL] Fallback SQL execution failed: {error_msg}. Last error: {str(last_error)}. Question ID: {query_id}")
            
            return fallback_sql 