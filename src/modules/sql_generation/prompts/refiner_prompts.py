"""SQL refiner module's prompt"""

REFINER_SYSTEM = """
You are a SQLite expert and very good at fixing SQL errors.
"""

REFINER_USER = """
### Instruction:
There is a SQL query generated based on the following Database Schema description and the potential Evidence to respond to the Question. 
However, executing this SQL has resulted in an error or empty result, and you need to fix it based on the error message. 
Utilize your knowledge of SQLite to generate the correct SQL. SQL must be surrounded by ```sql``` code block.

### Database schema:
{db_schema}

### Question:
{question} 

### Hint:
{evidence}

### SQL:
{sql} \n

### Execution result:
result_type: {result_type} 
result: {result} 
error_message: {error_message} \n

### Response:
""" 

REFINER_ALL_USER = """
You are a SQLite expert. 

### Instruction:
There is a SQL query generated based on the following Database Schema description and the potential Evidence to respond to the Question. 
However, executing this SQL has resulted in an error or empty result, and you need to fix it based on the error message. 
Utilize your knowledge of SQLite to generate the correct SQL.
If there is no problem, directly return the original SQL. SQL must be surrounded by ```sql``` code block.

### Database schema:
{db_schema}

### Question:
{question} 

### Hint:
{evidence}

### SQL:
{sql} \n

### Execution result:
result_type: {result_type} 
result: {result} 
error_message: {error_message} \n

### Response:
""" 