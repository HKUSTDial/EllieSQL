"""SQL refiner module's prompt"""

REFINER_SYSTEM = """
You are a SQLite expert. 

SQL must be surrounded by ```sql``` code block.
"""


REFINER_USER = """

There is a SQL query generated based on the following Database Schema description and the potential Evidence to respond to the Question. 
However, executing this SQL has resulted in an error, and you need to fix it based on the error message. 
Utilize your knowledge of SQLite to generate the correct SQL.
If there is no problem, directly return the original SQL.

Database schema:
{db_schema}

The question is:
{question} 

SQL:
{sql} \n
The execution result of this SQL is:
result_type: {result_type} 
result: {result} 
error_message: {error_message} \n

""" 