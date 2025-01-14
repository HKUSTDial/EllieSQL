"""Prompt template for the SQL generation module"""

SQL_GENERATION_SYSTEM = """
You are a SQLite SQL expert. 
"""


ONLINE_SYNTHESIS_PROMPT = """
You are a SQLite SQL expert. Your job is to create {k} examples, where each example consists of a question and a SQL query to fetch the data for it. 

### Example:
I want each example to look like this, question input and SQL output pairs:

"input": "What's the description of the series code SM.POP.TOTL for Aruba?
(Hints: Aruba is the name of the country where ShortName = 'Aruba')"

"output": "SELECT T2.Description FROM Country AS T1 INNER JOIN CountryNotes AS T2
ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = 'Aruba' AND T2.Seriescode = 'SM.POP.TOTL'"

### Instruction:
You should generate examples that examine and showcase different aspects and relationships of the following table schemas, 
described in "Table creation statements". Understand the database tables and their relationships. 
Understand the columns and their types and meanings to construct interesting examples.

Generate a mixture of SQL examples that include:
• some simple SQL query examples without JOIN
• some SQL query examples with aggregates, like COUNT
• some simple SQL query examples with JOIN
• some complex SQL query examples with nested JOIN

### Table creation statements
{TARGET_DATABASE_SCHEMA}

### Instruction:
Generate total of {k} examples. Only outputs the examples (question input and SQL output pairs), and each example can be separated by a new line.
""" 

ONLINE_SYNTHESIS_WITH_TRAIN = """
You are a SQLite SQL expert. Your job is to create a set of examples, where each example consists of a question and a SQL query to fetch the data for it.

### Instruction:
You should generate examples that examine and showcase different aspects and relationships of the following table schemas. 
Understand the database tables and their relationships. Understand the columns and their types and meanings to construct interesting examples.
I will also show you multiple examples generated for the other database and its table schemas, so you can see what kind of examples can be generated for a given database.

### Examples from other database：
The following is the table schemas and column examples for other database:
The database ({TRAIN_DATABASE_NAME}) structure is defined by the following table schemas
(comments after '-' provide additional column descriptions).
{TRAIN_DATABASE_SCHEMA}
————————
The folloiwing are the examples generated for the above database schemas:
Example 1) "input": "Among the countries in the group of Heavily Indebted Poor Countries, how many of them are under the lending category of the International Development Associations?
(Hints: group of Heavily Indebted Poor Countries is OtherGroups = 'HIPC'; International Development Associations refers to lendingcategory = 'IDA')"
"output": "SELECT COUNT(CountryCode) FROM Country WHERE LendingCategory = 'IDA' AND OtherGroups = 'HIPC'"
...
Example 10) "input": "What is the description of the footnote on the series code AG.LND.FRST.K2 in 1990 for Aruba?
(Hints: Year = 1990; Aruba is the name of country where ShortName = 'Aruba')"
"output": "SELECT T2.Description FROM Country AS T1 INNER JOIN FootNotes AS T2
ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = 'Aruba' AND T2.Seriescode = 'AG.LND.FRST.K2' AND T2.Year = 'YR1990'"

### Table creation statements
{TARGET_DATABASE_SCHEMA}

### Instruction:
Now similarly, generate examples (question input and SQL output pairs) for the table schemas defined below, in "Table creation statements".
Only outputs the examples (question input and SQL output pairs), and each example can be separated by a new line.
"""

