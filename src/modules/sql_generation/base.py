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
        生成SQL, 返回使用代码块包裹的生成SQL的raw output, 统一在generate_sql_with_retry中使用重试机制安全提取代码块中的SQL
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
                                    schema_linking_output = {
                                        "original_schema": database_schema,
                                        "linked_schema": enriched_linked_schema (如果使用enhanced linker, 则已经补充主键和外键, 而basic linker则无)
                                    }
            module_name: 模块名称
            
        Returns:
            str: 生成SQL的原始输出raw output, 供generate_sql_with_retry提取, 例如: Therefore, the final SQL is ```sql SELECT * FROM players;```
        """
        pass 
    
    async def generate_sql_with_retry(self, query: str, schema_linking_output: Dict, query_id: str, module_name: str = None) -> str:
        """
        带重试机制的SQL生成与提取, 仅用于防止无法提取出SQL的情况, 不进行validate_sql验证SQL的可执行性
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
            query_id: 查询ID
            module_name: 模块名称
            
        Returns:
            str: 生成的SQL（如果所有尝试都失败则返回fallback SQL）
        """
        last_error = None
        last_extracted_sql = None  # 保存最后一次成功提取的SQL
        
        for attempt in range(self.max_retries):
            try:
                raw_sql_output = await self.generate_sql(query, schema_linking_output, query_id, module_name)
                if raw_sql_output is not None:
                    extracted_sql = self.extractor.extract_sql(raw_sql_output)
                    if extracted_sql is not None:
                        return extracted_sql
                    else:
                        self.logger.warning(f"无法从输出中提取SQL, 第{attempt + 1}/{self.max_retries}次尝试")
                        continue
            except Exception as e:
                last_error = e
                self.logger.warning(f"SQL生成第{attempt + 1}/{self.max_retries}次尝试失败: {str(e)}")
                continue
                
        # 达到重试阈值，记录错误
        error_message = f"SQL生成在{self.max_retries}次尝试后完全失败，无法提取有效SQL。最后一次错误: {str(last_error)}。Question ID: {query_id} 程序继续执行..."
        self.logger.error(error_message)
        
        # 从schema中获取第一个表和它的第一个列作为fallback
        first_table = schema_linking_output["original_schema"]["tables"][0]
        table_name = first_table["table"]
        first_column = list(first_table["columns"].keys())[0]
        fallback_sql = f"SELECT {first_column} FROM {table_name} LIMIT 1;"
        
        return fallback_sql
    
    async def generate_sql_with_retry_with_validate(self, query: str, schema_linking_output: Dict, query_id: str, module_name: str = None) -> str:
        """
        NOTE: [deprecated]此函数已弃用, validate_sql应该由refiner实现
        带重试机制的SQL生成, 同时验证SQL的可执行性,达到重试阈值时返回最后一次有效的SQL, 如果完全失败则返回fallback SQL。
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
            query_id: 查询ID
            module_name: 模块名称
            
        Returns:
            str: 生成的SQL（如果所有尝试都失败则返回fallback SQL）
        """
        last_error = None
        last_extracted_sql = None  # 保存最后一次成功提取的SQL
        
        # 获取数据库路径
        if not self.data_file:
            raise ValueError("Data file path not set. Please call set_data_file() first.")
            
        data = load_json(self.data_file) # data_file: e.g. ./data/merge_dev_demo.json
        db_path = None
        
        for item in data:
            if item.get("question_id") == query_id:
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                # 使用与data_file相对的路径
                data_dir = os.path.dirname(self.data_file)
                db_folder = f"{source}_{db_id}"
                db_file = f"{db_id}.sqlite"
                db_path = str(Config().database_dir / db_folder / db_file)
                break
                
        if not db_path:
            raise ValueError(f"无法找到问题ID {query_id} 对应的数据库路径")
            
        for attempt in range(self.max_retries):
            try:
                raw_sql_output = await self.generate_sql(query, schema_linking_output, query_id, module_name)
                if raw_sql_output is not None:
                    extracted_sql = self.extractor.extract_sql(raw_sql_output)
                    if extracted_sql is not None:
                        last_extracted_sql = extracted_sql  # 更新最后一次成功提取的SQL
                        # 验证SQL是否可以执行且结果不为空
                        is_valid, error_msg = validate_sql_execution(db_path, extracted_sql)
                        if is_valid:
                            return extracted_sql
                        else:
                            self.logger.warning(f"SQL验证失败: {error_msg}")
                            continue
            except Exception as e:
                last_error = e
                self.logger.warning(f"SQL生成第{attempt + 1}/{self.max_retries}次尝试失败: {str(e)}")
                continue
                
        # 达到重试阈值，记录错误并返回最后一次有效的SQL
        if last_extracted_sql:
            error_message = f"SQL生成 共{self.max_retries}次尝试后失败，使用最后一次提取的SQL。 最后一次错误: {str(last_error)}。Question ID: {query_id} 程序继续执行..."
            self.logger.error(error_message)
            return last_extracted_sql
        else:
            # 完全失败时，生成一个简单的fallback SQL
            error_message = f"SQL生成 在{self.max_retries}次尝试后完全失败，无法提取有效SQL，使用fallback SQL。 最后一次错误: {str(last_error)}。Question ID: {query_id} 程序继续执行..."
            self.logger.error(error_message)
            
            # 从schema中获取第一个表和它的第一个列作为fallback，构造验证fallback SQL
            first_table = schema_linking_output["original_schema"]["tables"][0]
            table_name = first_table["table"]
            first_column = list(first_table["columns"].keys())[0]
            fallback_sql = f"SELECT {first_column} FROM {table_name} LIMIT 1;"
            is_valid, error_msg = validate_sql_execution(db_path, fallback_sql)
            
            if not is_valid:
                self.logger.fatal(f"[FATAL] Fallback SQL执行失败: {error_msg}。最后一次错误: {str(last_error)}。Question ID: {query_id}")
            
            return fallback_sql 