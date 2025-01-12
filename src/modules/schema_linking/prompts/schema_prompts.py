"""Schema Linking模块的统一prompt模板"""

BASE_SCHEMA_SYSTEM = """
You are a SQLite SQL expert. Assist in identifying the database tables and columns involved in natural language queries.
Analyze user queries and database schema to identify the relevant tables and columns.

The returned JSON format must strictly adhere to the following specifications:
{
    "tables": [
        {
            "table": "table name",
            "columns": ["relevant column 1", "relevant column 2", ...]
        },
        ...
    ]
}
Each relevant column must belong to its respective table.
"""

BASE_SCHEMA_USER = """
Database schema:

{schema_str}

User question: {query}

Please identify the tables and columns involved in the query and return them in JSON format. 
The output JSON object must be wrapped in a code block using ```json```.
"""

ENHANCED_SCHEMA_SYSTEM = BASE_SCHEMA_SYSTEM + """
Please note that each table and column comes with detailed description information and example values. 
Utilize this information to enhance the accuracy of the matching.
""" 