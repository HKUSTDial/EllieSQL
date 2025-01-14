"""Prompt template for the SQL generation module"""

SQL_GENERATION_SYSTEM = """You are a intelligent and responsible SQLite expert."""


SQL_GENERATION_USER = """
You are a intelligent and responsible SQLite expert. 

### Instruction:
You need to read and understand the following database schema description, as well as the evidence that may be used, and use your SQLite knowledge to generate SQL statements to answer user questions.
Follow these requirements:
- Return only the SQL statement, without any explanation.
- Use standard SQL syntax.
- Ensure the table and column names in the SQL statement exactly match the schema.
- Correctly join tables using the provided foreign key information.
- Outputted SQL must be surrounded by ```sql``` code block.

### Database Schema:
{schema}

### Hint:
{evidence}

### User Question:
{query}

Generate the corresponding SQL query surrounded by ```sql``` code block.
""" 