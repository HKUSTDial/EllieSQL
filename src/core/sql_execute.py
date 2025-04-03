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
            # Use URI and read-only mode to open the database, ensure read-only access to prevent the database from being tampered with during execution
            uri = f"file:{self.db_path}?mode=ro"
            with sqlite3.connect(uri, uri=True) as conn:
                cursor = conn.cursor()
                cursor.execute(self.sql)
                self.result = cursor.fetchall()
                cursor.close()  # Ensure the cursor is closed
        except Exception as e:
            self.exception = e
        finally:
            if conn:
                conn.close()  # Explicitly close the connection


def execute_sql_with_timeout(db_path: str, sql: str, timeout: int = 10) -> SQLExecutionResult:
    """
    Execute a SQL query synchronously with a timeout.
    
    :param db_path: The path to the database.
    :param sql: The SQL query to execute.
    :param timeout: The timeout.
    :return: The result of the SQL query.
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
    :param db_path: The path to the database.
    :param sql: The SQL query to check.
    :param with_limit: Whether to add a limit to the query.
    :param timeout: The timeout.
    :return: Whether the SQL query is valid.
    """
    if with_limit:
        sql = f"SELECT * FROM ({sql}) LIMIT 1;"
    return execute_sql_with_timeout(db_path, sql, timeout).result_type == SQLExecutionResultType.SUCCESS


def validate_sql_execution(db_path: str, sql: str) -> Tuple[bool, str]:
    """
    Check if a SQL query can be executed normally on a specified database and the result is not empty
    :param db_path: The path to the database file
    :param sql: The SQL query to validate
    :return: Tuple[bool, str]: (if valid, error message)
    """
    try:
        result = execute_sql_with_timeout(db_path, sql)
        # Check the execution result
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
    Check if a SQL query can be executed normally on a specified database and the result is not empty
    
    :param db_path: The path to the database file
    :param sql: The SQL query to validate
    :return: Tuple[bool, str]: (if valid, error message)
    """
    try:
        result = execute_sql_with_timeout(db_path, sql)
        # Check the execution result
        if result.result_type == SQLExecutionResultType.ERROR:
            return False, f"SQL execution error: {result.error_message}"
        elif result.result_type == SQLExecutionResultType.TIMEOUT:
            return False, "SQL execution timeout"
        elif result.result_type == SQLExecutionResultType.SUCCESS and (not result.result or len(result.result) == 0):
            return False, "SQL execution result is empty"
            
        return True, "SQL is executable"
        
    except Exception as e:
        return False, f"SQL validation exception: {str(e)}"