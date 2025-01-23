"""
Divide and Conquer Chain of Thought (CoT) Prompt for NL2SQL Tasks
"""


SQL_GENERATION_SYSTEM = """
You are a SQLite expert. 
"""


DIVIDE_PROMPT = """
You are a smart and responsible SQLite SQL expert. Given a database schema and a question, users want to know the corresponding SQL query. 
Your task is to understand the database schema and question, and decompose the question into sub-questions so user can better understand it. Each sub-question is enclosed in <<>>. Here is an example for reference:

### Example:
## Given the database schema:
Database: debit_card_specializing

Table name: customers
Columns:
  - CustomerID (INTEGER): Description: identification of the customer | Value examples: 3, 5, 6
  - Segment (TEXT): Meaning: client segment | Description: client segment | Value examples: SME, LAM, KAM
  - Currency (TEXT): Description: Currency | Value examples: EUR, CZK
Primary key: CustomerID

Table name: gasstations
Columns:
  - GasStationID (INTEGER): Meaning: Gas Station ID | Description: Gas Station ID | Value examples: 44, 45, 46
  - ChainID (INTEGER): Meaning: Chain ID | Description: Chain ID | Value examples: 13, 6, 23
  - Country (TEXT): Value examples: CZE, SVK
  - Segment (TEXT): Meaning: chain segment | Description: chain segment | Value examples: Value for money, Premium, Other
Primary key: GasStationID

Table name: products
Columns:
  - ProductID (INTEGER): Meaning: Product ID | Description: Product ID | Value examples: 1, 2, 3
  - Description (TEXT): Description: Description | Value examples: Rucní zadání, Nafta, Special
Primary key: ProductID

Table name: transactions_1k
Columns:
  - TransactionID (INTEGER): Meaning: Transaction ID | Description: Transaction ID | Value examples: 1, 2, 3
  - Date (DATE): Description: Date | Value examples: 2012-08-24, 2012-08-23, 2012-08-25
  - Time (TEXT): Description: Time | Value examples: 09:41:00, 10:03:00, 13:53:00
  - CustomerID (INTEGER): Meaning: Customer ID | Description: Customer ID | Value examples: 31543, 46707, 7654
  - CardID (INTEGER): Meaning: Card ID | Description: Card ID | Value examples: 486621, 550134, 684220
  - GasStationID (INTEGER): Meaning: Gas Station ID | Description: Gas Station ID | Value examples: 3704, 656, 741
  - ProductID (INTEGER): Meaning: Product ID | Description: Product ID | Value examples: 2, 23, 5
  - Amount (INTEGER): Description: Amount | Value examples: 28, 18, 1
  - Price (REAL): Description: Price | Value description: commonsense evidence:

total price = Amount x Price | Value examples: 672.64, 430.72, 121.99
Primary key: TransactionID

Table name: sqlite_sequence
Columns:
  - name (): Value examples: transactions_1k
  - seq (): Value examples: 1000

Table name: yearmonth
Columns:
  - CustomerID (INTEGER): Meaning: Customer ID | Description: Customer ID | Value examples: 39, 63, 172
  - Date (TEXT): Description: Date | Value examples: 201112, 201201, 201202
  - Consumption (REAL): Description: consumption | Value examples: 528.3, 1598.28, 1931.36
Primary key: CustomerID, Date

Foreign keys:
  yearmonth.CustomerID = customers.None
  yearmonth.CustomerID = customers.None

## Question: What is the gender of the youngest client who opened an account in the lowest average salary branch?

## Decompose the Question into sub-questions, each sub-question is enclosed in <<>>:

Sub-question 1: <<What is the lowest average salary branch?>>
Sub-question 2: <<Who is the youngest client in that branch?>>
Sub-question 3: <<What is the gender of that client?>>

### Your task: decompose the question into sub-questions.
## Given the database schema:
{schema}

## Question: 
{query}

## Hint:
{evidence}

## Decompose the Question into sub-questions, each sub-question is enclosed in <<>>:

""" 

CONQUER_PROMPT = """
You are a smart and responsible SQLite SQL expert. Given a database schema and a question, your task are:
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

### Examples:
{examples}

### Question: 
{query}

### Hint:
{evidence}

Please generate the corresponding SQL query. SQL must be surrounded by ```sql``` code block.
"""


CONQUER_PROMPT_WO_EXAMPLES = """
You are a smart and responsible SQLite SQL expert. Given a database schema and a question, your task are:
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

### Question: 
{query}

### Hint:
{evidence}

Please generate the corresponding SQL query. SQL must be surrounded by ```sql``` code block.
"""

ASSEMBLE_PROMPT = """
You are a smart and responsible SQLite SQL expert. Given a database schema and a question, users want to know the corresponding SQL query. 

### Instructions:
We have decomposed the main question into sub-questions, now your task is based on the SQL querys for corresponding sub-questions, assemble the final SQL for the main question:
1. Understand the database schema and the main question;
2. Read and analyze each sub-question and corresponding SQL query;
3. Analyze the relationship between sub-questions and the main question in order to assemble them properly;
4. Generate the final SQL for the main question and optimize it if needed.

### Database Schema:
{schema}

### Main question:
{query}

### Hint:
{evidence}

### Sub-questions and corresponding output, including SQL querys and explanation:
{subs}

Based on the SQL querys for corresponding sub-questions, generate the final SQL for the main question, SQL must be surrounded by ```sql``` code block.
"""