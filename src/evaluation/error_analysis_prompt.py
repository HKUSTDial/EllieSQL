"""Prompt for the SQL error analysis"""

ERROR_ANALYSIS_SYSTEM = """You are an SQLite expert, exceptionally adept at identifying errors in SQL."""

ERROR_ANALYSIS_USER = """
### Task Description:

Given that the query results of "generated_sql" and "golden_sql" on the database differ, you are tasked with identifying the errors in "generated_sql" and accurately categorizing them.
"question" represents the natural language query which the SQL should reflect, the "generated_sql" is the erroneous answer produced by the model, and the "golden_sql" is the reference standard answer.
"golden_schema" represents the relative database schema.

### question
{question}

### generated_sql
{generated_sql}

### golden_sql
{golden_sql}

### golden_schema
{golden_schema}


### Here are the error categories:

5. JOIN-Related Errors
Primarily failures related to JOIN operations.
- Main errors:
  - Missing essential tables (failure to recognize that the query requires JOINing multiple tables).
  - Incorrect foreign key joins (using incorrect foreign keys for JOIN operations).
  - Omission of necessary JOIN relationships, leading to incomplete query results.

7. GROUP BY-Related Errors
- Errors in this category mainly manifest as:
  - The query requires GROUP BY, but the model fails to recognize the grouping requirement.
  - Incorrect column selection in GROUP BY, leading to SQL syntax or logical errors.

11. Nested Query and Set Operation Errors
- This category mainly includes:
  - Missing nested queries (Nested Query), where the model fails to correctly generate subqueries.
  - Incorrect set operations (such as UNION, INTERSECT, EXCEPT), causing the query to fail to execute correctly.

13. Miscellaneous Errors
- Errors that cannot be classified into the above categories, mainly including:
  - Missing WHERE conditions (leading to incomplete query logic).
  - Superfluous or missing DISTINCT or DESC keywords.
  - Additional predicate conditions, causing the query results to not match the Gold SQL.

Return the final one error category surrounded by <<>> in the end of your response. 
For example, if the generated_sql missing an essential table(that requires JOIN), then your response should be <<5>>

""" 