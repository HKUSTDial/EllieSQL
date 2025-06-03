DIRECT_DC_PROMPT = """
You are an experienced database expert.
Now you need to generate a SQL query given the database information, a question and some additional information.
The database structure is defined by the following table schemas (comments after '--' provide additional column descriptions).
Note that the "Value Examples" are actual values from the column. Some column might contain the values that are directly related to the question. Use it to help you justify which columns to use.

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "recursive divide-and-conquer approach to SQL query generation from natural language".

Here is a high level description of the steps.
1. **Divide (Decompose Sub-question with Pseudo SQL):** The complex natural language question is recursively broken down into simpler sub-questions. Each sub-question targets a specific piece of information or logic required for the final SQL query. 
2. **Conquer (Real SQL for sub-questions):**  For each sub-question (and the main question initially), a "pseudo-SQL" fragment is formulated. This pseudo-SQL represents the intended SQL logic but might have placeholders for answers to the decomposed sub-questions. 
3. **Combine (Reassemble):** Once all sub-questions are resolved and their corresponding SQL fragments are generated, the process reverses. The SQL fragments are recursively combined by replacing the placeholders in the pseudo-SQL with the actual generated SQL from the lower levels.
4. **Final Output:** This bottom-up assembly culminates in the complete and correct SQL query that answers the original complex question.

Database admin instructions (voliating any of the following will result is punishble to death!):
1. **SELECT Clause:** 
    - Only select columns mentioned in the user's question. 
    - Avoid unnecessary columns or values.
2. **Aggregation (MAX/MIN):**
    - Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:**
    - Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC` to ensure distinct values.
4. **Handling NULLs:**
    - If a column may contain NULL values, use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:**
    - Only include tables essential to answer the question.
6. **Strictly Follow Hints:**
    - Adhere to all provided hints.
7. **Thorough Question Analysis:**
    - Address all conditions mentioned in the question.
8. **DISTINCT Keyword:**
    - Use `SELECT DISTINCT` when the question requires unique values (e.g., IDs, URLs). 
9. **Column Selection:**
    - Carefully analyze column descriptions and hints to choose the correct column when similar columns exist across tables.
10. **String Concatenation:**
    - Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause. 
11. **SQLite Functions Only:**
    - Use only functions available in SQLite.
12. **Date Processing:**
    - Utilize `STRFTIME()` for date manipulation (e.g., `STRFTIME('%Y', SOMETIME)` to extract the year).
13. **Schema Syntax:**
    - Use `table_name`.`column_name` to refer to columns from different tables, when table name or column name contains whitespace.
14. **JOIN Preference:**
    - Prioritize `INNER JOIN` over nested `SELECT` statements. Do not use `CROSS JOIN` or `LEFT/RIGHT JOIN`.

Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer, and finally try to simplify the SQL query using `INNER JOIN` over nested `SELECT` statements IF POSSIBLE.

Please respond with xml tags structured as follows:
```xml
<think>
Your detailed reasoning for the SQL query generation, with Recursive Divide-and-Conquer approach.
</think>
<sql>
The final SQL query that answers the question.
</sql>
```

**************************
【Table creation statements】
{SCHEMA_CONTEXT}

**************************
【Question】
Question: 
{QUESTION}

Hint:
{HINT}

**************************

Only output xml format (starting with ```xml and ending with ```) as your response.

"""





DIRECT_DC_WITH_OS_PROMPT = """
You are an experienced database expert.
Now you need to generate a SQL query given the database information, a question and some additional information.
The database structure is defined by the following table schemas (comments after '--' provide additional column descriptions).
Note that the "Value Examples" are actual values from the column. Some column might contain the values that are directly related to the question. Use it to help you justify which columns to use.

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "recursive divide-and-conquer approach to SQL query generation from natural language".

Here is a high level description of the steps.
1. **Divide (Decompose Sub-question with Pseudo SQL):** The complex natural language question is recursively broken down into simpler sub-questions. Each sub-question targets a specific piece of information or logic required for the final SQL query. 
2. **Conquer (Real SQL for sub-questions):**  For each sub-question (and the main question initially), a "pseudo-SQL" fragment is formulated. This pseudo-SQL represents the intended SQL logic but might have placeholders for answers to the decomposed sub-questions. 
3. **Combine (Reassemble):** Once all sub-questions are resolved and their corresponding SQL fragments are generated, the process reverses. The SQL fragments are recursively combined by replacing the placeholders in the pseudo-SQL with the actual generated SQL from the lower levels.
4. **Final Output:** This bottom-up assembly culminates in the complete and correct SQL query that answers the original complex question.

Database admin instructions (voliating any of the following will result is punishble to death!):
1. **SELECT Clause:** 
    - Only select columns mentioned in the user's question. 
    - Avoid unnecessary columns or values.
2. **Aggregation (MAX/MIN):**
    - Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:**
    - Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC` to ensure distinct values.
4. **Handling NULLs:**
    - If a column may contain NULL values, use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:**
    - Only include tables essential to answer the question.
6. **Strictly Follow Hints:**
    - Adhere to all provided hints.
7. **Thorough Question Analysis:**
    - Address all conditions mentioned in the question.
8. **DISTINCT Keyword:**
    - Use `SELECT DISTINCT` when the question requires unique values (e.g., IDs, URLs). 
9. **Column Selection:**
    - Carefully analyze column descriptions and hints to choose the correct column when similar columns exist across tables.
10. **String Concatenation:**
    - Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause. 
11. **SQLite Functions Only:**
    - Use only functions available in SQLite.
12. **Date Processing:**
    - Utilize `STRFTIME()` for date manipulation (e.g., `STRFTIME('%Y', SOMETIME)` to extract the year).
13. **Schema Syntax:**
    - Use `table_name`.`column_name` to refer to columns from different tables, when table name or column name contains whitespace.
14. **JOIN Preference:**
    - Prioritize `INNER JOIN` over nested `SELECT` statements. Do not use `CROSS JOIN` or `LEFT/RIGHT JOIN`.

Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer, and finally try to simplify the SQL query using `INNER JOIN` over nested `SELECT` statements IF POSSIBLE.

Here are some NL2SQL examples for your reference:
{OS_EXAMPLES}

Please respond with xml tags structured as follows:
```xml
<think>
Your detailed reasoning for the SQL query generation, with Recursive Divide-and-Conquer approach.
</think>
<sql>
The final SQL query that answers the question.
</sql>
```

**************************
【Table creation statements】
{SCHEMA_CONTEXT}

**************************
【Question】
Question: 
{QUESTION}

Hint:
{HINT}

**************************

Only output xml format (starting with ```xml and ending with ```) as your response.

"""