"""
Divide and Conquer Chain of Thought (CoT) Prompt for NL2SQL Tasks
"""


SQL_GENERATION_SYSTEM = """
You are a SQLite expert. 
Outputted SQL must be surrounded by ```sql``` code block.
"""


DIVIDE_PROMPT = """
### Example:
## Given the database schema:
数据库: debit_card_specializing

表名: customers
列:
  - CustomerID (INTEGER): 描述: identification of the customer | 示例值: 3, 5, 6
  - Segment (TEXT): 含义: client segment | 描述: client segment | 示例值: SME, LAM, KAM
  - Currency (TEXT): 描述: Currency | 示例值: EUR, CZK
主键: CustomerID

表名: gasstations
列:
  - GasStationID (INTEGER): 含义: Gas Station ID | 描述: Gas Station ID | 示例值: 44, 45, 46
  - ChainID (INTEGER): 含义: Chain ID | 描述: Chain ID | 示例值: 13, 6, 23
  - Country (TEXT): 示例值: CZE, SVK
  - Segment (TEXT): 含义: chain segment | 描述: chain segment | 示例值: Value for money, Premium, Other
主键: GasStationID

表名: products
列:
  - ProductID (INTEGER): 含义: Product ID | 描述: Product ID | 示例值: 1, 2, 3
  - Description (TEXT): 描述: Description | 示例值: Rucní zadání, Nafta, Special
主键: ProductID

表名: transactions_1k
列:
  - TransactionID (INTEGER): 含义: Transaction ID | 描述: Transaction ID | 示例值: 1, 2, 3
  - Date (DATE): 描述: Date | 示例值: 2012-08-24, 2012-08-23, 2012-08-25
  - Time (TEXT): 描述: Time | 示例值: 09:41:00, 10:03:00, 13:53:00
  - CustomerID (INTEGER): 含义: Customer ID | 描述: Customer ID | 示例值: 31543, 46707, 7654
  - CardID (INTEGER): 含义: Card ID | 描述: Card ID | 示例值: 486621, 550134, 684220
  - GasStationID (INTEGER): 含义: Gas Station ID | 描述: Gas Station ID | 示例值: 3704, 656, 741
  - ProductID (INTEGER): 含义: Product ID | 描述: Product ID | 示例值: 2, 23, 5
  - Amount (INTEGER): 描述: Amount | 示例值: 28, 18, 1
  - Price (REAL): 描述: Price | 值描述: commonsense evidence:

total price = Amount x Price | 示例值: 672.64, 430.72, 121.99
主键: TransactionID

表名: sqlite_sequence
列:
  - name (): 示例值: transactions_1k
  - seq (): 示例值: 1000

表名: yearmonth
列:
  - CustomerID (INTEGER): 含义: Customer ID | 描述: Customer ID | 示例值: 39, 63, 172
  - Date (TEXT): 描述: Date | 示例值: 201112, 201201, 201202
  - Consumption (REAL): 描述: consumption | 示例值: 528.3, 1598.28, 1931.36
主键: CustomerID, Date

外键关系:
  yearmonth.CustomerID = customers.None
  yearmonth.CustomerID = customers.None

## Question: What is the gender of the youngest client who opened an account in the lowest average salary branch?

## Decompose the Question into sub-questions, each sub-question is enclosed in <<>>:

<<Sub-question 1: What is the lowest average salary branch?>>
<<Sub-question 2: Who is the youngest client in that branch?>>
<<Sub-question 3: What is the gender of that client?>>

### Now do the following task:
## Given the database schema:
{schema}

## Question: 
{query}

## Decompose the Question into sub-questions, each sub-question is enclosed in <<>>:

""" 

CONQUER_PROMPT = """
Given the database schema:
{schema}

Examples:
{examples}

Question: 
{query}

Please generate the corresponding SQL query. SQL must be surrounded by ```sql``` code block.
"""

ASSEMBLE_PROMPT = """
Given the database schema:
{schema}

Main question:
{query}
Based on the sql querys for corresponding sub-questions, return the final sql for the main question, SQL must be surrounded by ```sql``` code block.

Sub-questions and corresponding sql querys:
{subs}
"""