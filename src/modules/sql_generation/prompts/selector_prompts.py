"""SQL selector 模块的 prompt"""

SELECTOR_SYSTEM = """
You are a SQLite expert. 
SQL must be surrounded by ```sql``` code block.
"""

SELECTOR_USER = """
Regarding the Question, there are {candidate_num} candidate SQL along with their Execution result in the database (showing the first 10 rows).
You need to compare these candidates and analyze the differences among the various candidate SQL. 
Based on the provided Database Schema, Evidence, and Question, select the correct and reasonable result.

[Database Schema]
{db_schema}

[Evidence]
{evidence}

[Question]
{question}

==========

Candidates:

""" 

CANDIDATE_FORMAT = """
Candidate {i}
[SQL]
{isql}
[Execution result]
result_type: {iresult_type} 
result: {iresult} 
error_message: {ierror_message} \n

********
"""

