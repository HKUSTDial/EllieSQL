"""Prompt template for the SQL generation module"""

SQL_GENERATION_SYSTEM = """You are a SQLite expert."""

SQL_GENERATION_USER = """
You are a intelligent and responsible SQLite expert. 

### Instruction:
You need to read the database schema to generate SQL query for the user question. The outputted SQL must be surrounded by ```sql``` code block.

### Database Schema:
{schema}

### Hint:
{evidence}

### User Question:
{query}

Generate the corresponding SQL query surrounded by ```sql``` code block.
""" 

SQL_GENERATION_USER_OLD = """
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

SQL_GENERATION_USER_OLD_2 = """
You are a smart and responsible SQLite SQL expert. Given a database schema and a question, users want to know the corresponding SQL query. 
1. Parse user questions:
  - Use natural language processing (NLP) techniques to parse user questions and extract query requirements and conditions.

2. Analyze database schema:
  - Based on the database schema, understand the fields and relationships of the table, and build the basic framework of the SQL query.

3. Check sample data:
  - Analyze the data characteristics based on the first three rows of the table values to help determine how to construct query conditions and filter results.

4. Generate SQL query:
  - Based on user questions, query requirements and conditions, database schema, and sample data, build a complete SQL query.

5. Verification and optimization:
  - Check whether the generated SQL query is logical and optimize it if necessary.

### Database Schema:
{schema}

### Hint:
{evidence}

### User Question:
{query}

Generate the corresponding SQL query surrounded by ```sql``` code block.
"""