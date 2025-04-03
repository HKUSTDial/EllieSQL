"""SQL post-processing module prompt templates"""

REFLECTION_SYSTEM = """You are a SQL expert responsible for checking and optimizing SQL statements.
Check the following points:
1. whether the SQL syntax is correct
2. whether the join conditions of tables are reasonable
3. whether the conditions of WHERE clause are appropriate or not

If a problem is found, return the corrected SQL directly If there is no problem, return the original SQL The SQL statement must be wrapped in ```sql```."""

REFLECTION_USER = """Please check and optimize the following SQL:
{sql}""" 


FEEDBACK_BASED_REFLECTION_USER = """Please check and optimize the following SQL based on the SQL execution result and natural language expression, ensuring that the SQL correctly reflects the intention of the natural language expression:
{sql} \n
The SQL execution result is:
result_type: {result_type} \n
result: {result} \n
error_message: {error_message} \n
The original natural language expression is:
{question}
""" 