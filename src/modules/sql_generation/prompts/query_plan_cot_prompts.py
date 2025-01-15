"""
Query Plan Chain of Thought (CoT) Prompt for NL2SQL Tasks
"""

SQL_GENERATION_SYSTEM = """You are a intelligent and responsible SQLite expert. The Generated corresponding SQL query must be surrounded by ```sql``` code block."""


QUERY_PLAN_PROMPT = """
Database Info
{schema}
**************************
Answer Repeating the question and evidence, and generating the SQL with a query plan.
**Question**: {query}
**Evidence**: {evidence}

**Query Plan**:
** Preparation Steps:**
1. Initialize the process: Start preparing to execute the query.
2. Prepare storage: Set up storage space (registers) to hold temporary results, initializing them to NULL.
3. Open the location table: Open the location table so we can read from it.
4. Open the generalinfo table: Open the generalinfo table so we can read from it.
"""

