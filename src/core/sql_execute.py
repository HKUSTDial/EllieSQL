import sqlite3
import threading
from enum import Enum
from typing import Optional, List, Tuple


class SQLExecutionResultType(Enum):
    """
    Type of the result of a SQL query execution.
    
    Attributes:
        SUCCESS: The SQL query is executed successfully.
        TIMEOUT: The SQL query execution timed out.
        ERROR: The SQL query execution failed.
    """
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    
class SQLExecutionResult:
    """
    Result of a SQL query execution.
    """
    def __init__(self, db_path: str, sql: str, result_type: SQLExecutionResultType, result: Optional[List[Tuple]], error_message: Optional[str]) -> None:
        self.db_path = db_path
        self.sql = sql
        self.result_type = result_type
        self.result = result
        self.error_message = error_message
        
    def to_dict(self) -> dict:
        return {
            "db_path": self.db_path,
            "sql": self.sql,
            "result_type": self.result_type.value,
            "result": self.result,
            "error_message": self.error_message
        }
    

class ExecuteSQLThread(threading.Thread):
    """
    Thread to execute a SQL query.
    """
    def __init__(self, db_path: str, sql: str, timeout: int) -> None:
        super().__init__()
        self.db_path = db_path
        self.sql = sql
        self.timeout = timeout
        self.result = None
        self.exception = None
        
    def run(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(self.sql)
                self.result = cursor.fetchall()
                cursor.close()  # 确保关闭游标
        except Exception as e:
            self.exception = e
        finally:
            if conn:
                conn.close()  # 显式关闭连接


def execute_sql_with_timeout(db_path: str, sql: str, timeout: int = 10) -> SQLExecutionResult:
    """
    Execute a SQL query synchronously with a timeout.
    
    Args:
        db_path: The path to the database.
        sql: The SQL query to execute.
        timeout: The timeout.
    Returns:
        The result of the SQL query.
    """ 
    thread = ExecuteSQLThread(db_path, sql, timeout)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        error_message = f"SQL execution timed out after {timeout} seconds"
        return SQLExecutionResult(db_path, sql, SQLExecutionResultType.TIMEOUT, None, error_message)
    if thread.exception:
        error_message = str(thread.exception)
        return SQLExecutionResult(db_path, sql, SQLExecutionResultType.ERROR, None, error_message)
    return SQLExecutionResult(db_path, sql, SQLExecutionResultType.SUCCESS, thread.result, None)


def check_query_valid(db_path: str, sql: str, with_limit: bool = True, timeout: int = 10) -> bool:
    """
    Check if a SQL query is valid.
    
    Args:
        db_path: The path to the database.
        sql: The SQL query to check.
        with_limit: Whether to add a limit to the query.
        timeout: The timeout.
    Returns:
        Whether the SQL query is valid.
    """
    if with_limit:
        sql = f"SELECT * FROM ({sql}) LIMIT 1;"
    return execute_sql_with_timeout(db_path, sql, timeout).result_type == SQLExecutionResultType.SUCCESS


def validate_sql_execution(db_path: str, sql: str) -> Tuple[bool, str]:
    """
    验证SQL是否可以在指定数据库上正常执行且结果不为空
    
    Args:
        db_path: 数据库文件路径
        sql: 待验证的SQL语句
        
    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    try:
        result = execute_sql_with_timeout(db_path, sql)
        # 检查执行结果
        if result.result_type == SQLExecutionResultType.ERROR:
            return False, f"SQL execution error: {result.error_message}"
        elif result.result_type == SQLExecutionResultType.TIMEOUT:
            return False, "SQL execution timeout"
        # elif result.result_type == SQLExecutionResultType.SUCCESS and (not result.result or len(result.result) == 0):
        #     return False, "SQL execution result is empty"
            
        return True, "SQL is executable"
        
    except Exception as e:
        return False, f"SQL validation exception: {str(e)}"



def validate_sql_execution_and_non_empty_result(db_path: str, sql: str) -> Tuple[bool, str]:
    """
    验证SQL是否可以在指定数据库上正常执行且结果不为空
    
    Args:
        db_path: 数据库文件路径
        sql: 待验证的SQL语句
        
    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    try:
        result = execute_sql_with_timeout(db_path, sql)
        # 检查执行结果
        if result.result_type == SQLExecutionResultType.ERROR:
            return False, f"SQL execution error: {result.error_message}"
        elif result.result_type == SQLExecutionResultType.TIMEOUT:
            return False, "SQL execution timeout"
        elif result.result_type == SQLExecutionResultType.SUCCESS and (not result.result or len(result.result) == 0):
            return False, "SQL execution result is empty"
            
        return True, "SQL is executable"
        
    except Exception as e:
        return False, f"SQL validation exception: {str(e)}"