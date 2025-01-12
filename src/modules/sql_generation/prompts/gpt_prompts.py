"""Prompt template for the SQL generation module"""

SQL_GENERATION_SYSTEM = """You are a SQLite expert. Generate accurate SQL statements based on the user's natural language query.
Follow these rules:
Return only the SQL statement, without any explanation.
Use standard SQL syntax.
Ensure the table and column names in the SQL statement exactly match the schema.
Correctly join tables using the provided foreign key information.
Outputted SQL must be surrounded by ```sql``` code block.
"""

# SQL_GENERATION_USER = """
# 完整的数据库Schema:
# {schema_str}

# 根据Schema Linking识别出的相关表和列:
# 相关表: {tables}
# 相关列: {columns}

# 用户查询: {query}

# 请生成SQL语句:""" 


SQL_GENERATION_USER = """
Given the database schema:
{schema}

User question: 
{query}

Please generate the corresponding SQL query. SQL must be surrounded by ```sql``` code block.
""" 