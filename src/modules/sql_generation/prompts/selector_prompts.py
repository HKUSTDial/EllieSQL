"""SQL selector module's prompt"""

SELECTOR_SYSTEM = """
You are a SQLite expert. 
"""

SELECTOR_USER = """
You are a SQLite expert. 

### Instruction:
Regarding the Question, there are {candidate_num} candidate SQL along with their Execution result in the database (showing the first 10 rows).
You need to compare these candidates and analyze the differences among the various candidate SQL. 
Based on the provided Database Schema, Evidence, and Question, select the correct and reasonable candidate SQL, which must be surrounded by ```sql``` code block.

### Database Schema:
{db_schema}

### Hint:
{evidence}

### Question:
{question}

### Candidates:
{candidate_list}

### Response:
Think carefully and select the correct candidate SQL, which must be surrounded by ```sql``` code block.
""" 

CANDIDATE_FORMAT = """
Candidate {i}:
[SQL]
{isql}
[Execution result]
result_type: {iresult_type} 
result: {iresult} 
error_message: {ierror_message} \n

********
"""

