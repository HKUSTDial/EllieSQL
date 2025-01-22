"""
Query Plan Chain of Thought (CoT) Prompt for NL2SQL Tasks
"""

SQL_GENERATION_SYSTEM = """You are a SQLite expert."""


QUERY_PLAN_PROMPT = """
### Instruction:
You need to read and understand the following database schema description, as well as the evidence that may be used, and use your SQLite knowledge to generate SQL statements to answer user questions.
You should try to generate a query plan in order to make sure the generated SQL query is correct and align with user question. The Generated corresponding SQL query must be surrounded by ```sql``` code block.

1. Preparation;
2. Understand and analyze database schema and question, identify and locate the relevant tables for the question;
3. Perform operations such as counting, filtering, or matching between tables if needed;
4. Deliver the final SQL query and optimize if needed.

Answer Repeating the question and evidence, and generating the SQL with a query plan.

### Database Schema:
{schema}

### Question:
# {query}

### Hint: 
{evidence}

### Query Plan:
## Preparation Steps:
1. Initialize the process: Start preparing to execute the query.
2. Prepare storage: Set up storage space (registers) to hold temporary results, initializing them to NULL.
3. Open the location table: Open the location table so we can read from it.
4. Open the generalinfo table: Open the generalinfo table so we can read from it.
"""

QUERY_PLAN_GEN = """
Database Info
{schema}
**************************
Generating the a query plan for sql generation.
**Question**: {query}
**Evidence**: {evidence}
"""

SQL_GEN = """
### Generate the corresponding SQL of the question with the Query Plan:
Database Info
{schema}

**Question**: {query}
**Evidence**: {evidence}
**Query Plan**:
{query_plan}

"""


QUERY_PLAN_PROMPT2 = """
### Generate the corresponding SQL of the question with Query Plan, here is an example:
{example}

### Your task:
Database Info
{schema}
**************************
Answer Repeating the question and evidence, and generating the SQL with a query plan.
**Question**: {query}
**Evidence**: {evidence}

"""

EXAMPLE = """
Database Info
Database: restaurant

Table name: geographic
Columns:
  - city (TEXT): Description: the city | Value examples: alameda, alamo, albany
  - county (TEXT): Meaning: country | Description: the country the city belongs to | Value examples: alameda county, contra costa county, santa clara county
  - region (TEXT): Description: corresponding regions | Value examples: bay area, napa valley, unknown
Primary key: city

Table name: generalinfo
Columns:
  - id_restaurant (INTEGER): Meaning: id restaurant | Description: the unique id for the restaurant | Value examples: 1, 2, 3
  - label (TEXT): Description: the label of the restaurant | Value examples: sparky's diner, kabul afghan cuisine, helmand restaurant
  - food_type (TEXT): Meaning: food type | Description: the food type | Value examples: 24 hour diner, afghani, african
  - city (TEXT): Description: the city where the restaurant is located in | Value examples: san francisco, san carlos, sunnyvale
  - review (REAL): Description: the review of the restaurant | Value description: commonsense evidence:
the review rating is from 0.0 to 5.0
The high review rating is positively correlated with the overall level of the restaurant. The restaurant with higher review rating is usually more popular among diners. | Value examples: 2.3, 3.8, 4.0
Primary key: id_restaurant

Table name: location
Columns:
  - id_restaurant (INTEGER): Meaning: id restaurant | Description: the unique id for the restaurant | Value examples: 1, 2, 3
  - street_num (INTEGER): Meaning: street number | Description: the street number of the restaurant | Value examples: 242, 135, 430
  - street_name (TEXT): Meaning: street name | Description: the street name of the restaurant | Value examples: church st, el camino real, broadway   
  - city (TEXT): Description: the city where the restaurant is located in | Value examples: san francisco, san carlos, sunnyvale
Primary key: id_restaurant

Foreign keys:
  generalinfo.city = geographic.city
  location.id_restaurant = generalinfo.id_restaurant
  location.city = geographic.city
**************************
Answer Repeating the question and evidence, and generating the SQL with a query plan.
**Question**: How many Thai restaurants can be found in San Pablo Ave, Albany?

**Evidence**: Thai restaurant refers to food type = ’thai’; San Pablo Ave Albany refers to street name = ’san pablo ave’ AND T1.city = ’albany’

**Query Plan**:
** Preparation Steps:**
1. Initialize the process: Start preparing to execute the query.
2. Prepare storage: Set up storage space (registers) to hold temporary results, initializing them to NULL.
3. Open the location table: Open the location table so we can read from it.
4. Open the generalinfo table: Open the generalinfo table so we can read from it.

** Matching Restaurants:**
1. Start reading the location table: Move to the first row in the location table.
2. Check if the street matches: Look at the street name column of the current row in location. If it’s not ”san pablo ave,” skip this row.
3. Identify the matching row: Store the identifier (row ID) of this location entry.
4. Find the corresponding row in generalinfo: Use the row ID from location to directly find the matching row in generalinfo.
5. Check if the food type matches: Look at the food type column in generalinfo. If it’s not ”thai,” skip this row.
6. Check if the city matches: Look at the city column in generalinfo. If it’s not ”albany,” skip this row.

** Counting Restaurants:**
1. Prepare to count this match: If all checks pass, prepare to include this row in the final count.
2. Count this match: Increment the count for each row that meets all the criteria.
3. Move to the next row in location: Go back to the location table and move to the next row, repeating the process until all rows are checked.
4. Finalize the count: Once all rows have been checked, finalize the count of matching rows.
5. Prepare the result: Copy the final count to prepare it for output.

** Delivering the Result:**
1. Output the result: Output the final count, which is the number of restaurants that match all the specified criteria.
2. End the process: Stop the query execution process.
3. Setup phase: Before starting the actual query execution, the system prepares the specific values it will be looking for, like ”san pablo ave,” ”thai,” and ”albany.”

**Final Optimized SQL Query:**
```sql
SELECT COUNT(T1.id restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id restaurant = T2.id restaurant WHERE T1.food type = ’thai’ AND T1.city = ’albany’ AND T2.street name = ’san pablo ave’
```
"""

