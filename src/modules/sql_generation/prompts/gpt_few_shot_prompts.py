"""Prompt template for the SQL generation module"""

SQL_GENERATION_SYSTEM = """You are a intelligent and responsible SQLite expert."""


SQL_GENERATION_USER = """
You are a intelligent and responsible SQLite expert. 

### Instruction:
You need to read and understand the following database schema description, as well as the evidence that may be used, and use your SQLite knowledge to generate SQL statements to answer user questions.
Follow these requirements:
- Return only the SQL statement, without any explanation.
- Use standard SQL syntax.
- Ensure the table and column names in the SQL statement exactly match the schema.
- Correctly join tables using the provided foreign key information.
- Outputted SQL must be surrounded by ```sql``` code block.

### Examples:
{examples}
Now let's start!

### Database Schema:
{schema}

### Hint:
{evidence}

### User Question:
{query}

Generate the corresponding SQL query surrounded by ```sql``` code block.
""" 

EXAMPLES = """
Examples:
Easy example 1:
Q: "Find the buildings which have rooms with capacity more than 50."
Database: college_2

Table name: classroom
Columns:
  - building (varchar(15)): Value examples: Alumni, Bronfman, Chandler
  - room_number (varchar(7)): Value examples: 143, 547, 700
  - capacity (numeric(4,0)): Value examples: 10, 27, 92
Primary key: building, room_number

Table name: department
Columns:
  - dept_name (varchar(20)): Value examples: Accounting, Astronomy, Athletics
  - building (varchar(15)): Value examples: Chandler, Candlestick, Taylor
  - budget (numeric(12,2)): Value examples: 255041.46, 647610.55, 699140.86
Primary key: dept_name

Table name: course
Columns:
  - course_id (varchar(8)): Value examples: 101, 105, 123
  - title (varchar(50)): Value examples: C  Programming, The Music of Donovan, Electron Microscopy
  - dept_name (varchar(20)): Value examples: Mech. Eng., Comp. Sci., Statistics
  - credits (numeric(2,0)): Value examples: 4, 3
Primary key: course_id

Table name: instructor
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 16807
  - name (varchar(20)): Value examples: McKinnon, Pingr, Mird
  - dept_name (varchar(20)): Value examples: Cybernetics, Statistics, Marketing
  - salary (numeric(8,2)): Value examples: 94333.99, 59303.62, 119921.41
Primary key: ID

Table name: section
Columns:
  - course_id (varchar(8)): Value examples: 105, 137, 158
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2009, 2002, 2008
  - building (varchar(15)): Value examples: Chandler, Gates, Whitman
  - room_number (varchar(7)): Value examples: 804, 314, 434
  - time_slot_id (varchar(4)): Value examples: N, K, O
Primary key: course_id, sec_id, semester, year

Table name: teaches
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 19368
  - course_id (varchar(8)): Value examples: 747, 169, 445
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Spring, Fall
  - year (numeric(4,0)): Value examples: 2004, 2007, 2001
Primary key: ID, course_id, sec_id, semester, year

Table name: student
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - name (varchar(20)): Value examples: Schrefl, Rumat, Miliko
  - dept_name (varchar(20)): Value examples: History, Finance, Statistics
  - tot_cred (numeric(3,0)): Value examples: 4, 100, 116
Primary key: ID

Table name: takes
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - course_id (varchar(8)): Value examples: 239, 319, 362
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2006, 2003, 2005
  - grade (varchar(2)): Value examples: C-, B-, A-
Primary key: ID, course_id, sec_id, semester, year

Table name: advisor
Columns:
  - s_ID (varchar(5)): Value examples: 1000, 10033, 10076
  - i_ID (varchar(5)): Value examples: 19368, 95030, 22591
Primary key: s_ID

Table name: time_slot
Columns:
  - time_slot_id (varchar(4)): Value examples: A, B, C
  - day (varchar(1)): Value examples: F, M, W
  - start_hr (numeric(2)): Value examples: 8, 9, 11
  - start_min (numeric(2)): Value examples: 0, 30
  - end_hr (numeric(2)): Value examples: 8, 9, 11
  - end_min (numeric(2)): Value examples: 50, 45, 30
Primary key: time_slot_id, day, start_hr, start_min

Table name: prereq
Columns:
  - course_id (varchar(8)): Value examples: 133, 158, 169
  - prereq_id (varchar(8)): Value examples: 130, 169, 345
Primary key: course_id, prereq_id

Foreign keys:
  course.dept_name = department.dept_name
  instructor.dept_name = department.dept_name
  section.building = classroom.building
  section.room_number = classroom.room_number
  section.course_id = course.course_id
  teaches.ID = instructor.ID
  teaches.course_id = section.course_id
  teaches.sec_id = section.sec_id
  teaches.semester = section.semester
  teaches.year = section.year
  student.dept_name = department.dept_name
  takes.ID = student.ID
  takes.course_id = section.course_id
  takes.sec_id = section.sec_id
  takes.semester = section.semester
  takes.year = section.year
  advisor.s_ID = student.ID
  advisor.i_ID = instructor.ID
  prereq.prereq_id = course.course_id
  prereq.course_id = course.course_id
SQL: SELECT DISTINCT building FROM classroom WHERE capacity  >  50

Easy example 2:
Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
Database: college_2

Table name: classroom
Columns:
  - building (varchar(15)): Value examples: Alumni, Bronfman, Chandler
  - room_number (varchar(7)): Value examples: 143, 547, 700
  - capacity (numeric(4,0)): Value examples: 10, 27, 92
Primary key: building, room_number

Table name: department
Columns:
  - dept_name (varchar(20)): Value examples: Accounting, Astronomy, Athletics
  - building (varchar(15)): Value examples: Chandler, Candlestick, Taylor
  - budget (numeric(12,2)): Value examples: 255041.46, 647610.55, 699140.86
Primary key: dept_name

Table name: course
Columns:
  - course_id (varchar(8)): Value examples: 101, 105, 123
  - title (varchar(50)): Value examples: C  Programming, The Music of Donovan, Electron Microscopy
  - dept_name (varchar(20)): Value examples: Mech. Eng., Comp. Sci., Statistics
  - credits (numeric(2,0)): Value examples: 4, 3
Primary key: course_id

Table name: instructor
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 16807
  - name (varchar(20)): Value examples: McKinnon, Pingr, Mird
  - dept_name (varchar(20)): Value examples: Cybernetics, Statistics, Marketing
  - salary (numeric(8,2)): Value examples: 94333.99, 59303.62, 119921.41
Primary key: ID

Table name: section
Columns:
  - course_id (varchar(8)): Value examples: 105, 137, 158
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2009, 2002, 2008
  - building (varchar(15)): Value examples: Chandler, Gates, Whitman
  - room_number (varchar(7)): Value examples: 804, 314, 434
  - time_slot_id (varchar(4)): Value examples: N, K, O
Primary key: course_id, sec_id, semester, year

Table name: teaches
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 19368
  - course_id (varchar(8)): Value examples: 747, 169, 445
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Spring, Fall
  - year (numeric(4,0)): Value examples: 2004, 2007, 2001
Primary key: ID, course_id, sec_id, semester, year

Table name: student
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - name (varchar(20)): Value examples: Schrefl, Rumat, Miliko
  - dept_name (varchar(20)): Value examples: History, Finance, Statistics
  - tot_cred (numeric(3,0)): Value examples: 4, 100, 116
Primary key: ID

Table name: takes
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - course_id (varchar(8)): Value examples: 239, 319, 362
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2006, 2003, 2005
  - grade (varchar(2)): Value examples: C-, B-, A-
Primary key: ID, course_id, sec_id, semester, year

Table name: advisor
Columns:
  - s_ID (varchar(5)): Value examples: 1000, 10033, 10076
  - i_ID (varchar(5)): Value examples: 19368, 95030, 22591
Primary key: s_ID

Table name: time_slot
Columns:
  - time_slot_id (varchar(4)): Value examples: A, B, C
  - day (varchar(1)): Value examples: F, M, W
  - start_hr (numeric(2)): Value examples: 8, 9, 11
  - start_min (numeric(2)): Value examples: 0, 30
  - end_hr (numeric(2)): Value examples: 8, 9, 11
  - end_min (numeric(2)): Value examples: 50, 45, 30
Primary key: time_slot_id, day, start_hr, start_min

Table name: prereq
Columns:
  - course_id (varchar(8)): Value examples: 133, 158, 169
  - prereq_id (varchar(8)): Value examples: 130, 169, 345
Primary key: course_id, prereq_id

Foreign keys:
  course.dept_name = department.dept_name
  instructor.dept_name = department.dept_name
  section.building = classroom.building
  section.room_number = classroom.room_number
  section.course_id = course.course_id
  teaches.ID = instructor.ID
  teaches.course_id = section.course_id
  teaches.sec_id = section.sec_id
  teaches.semester = section.semester
  teaches.year = section.year
  student.dept_name = department.dept_name
  takes.ID = student.ID
  takes.course_id = section.course_id
  takes.sec_id = section.sec_id
  takes.semester = section.semester
  takes.year = section.year
  advisor.s_ID = student.ID
  advisor.i_ID = instructor.ID
  prereq.prereq_id = course.course_id
  prereq.course_id = course.course_id
SQL: SELECT building ,  room_number FROM classroom WHERE capacity BETWEEN 50 AND 100

Medium example 1:
Q: "Find the total budgets of the Marketing or Finance department."
Database: college_2

Table name: classroom
Columns:
  - building (varchar(15)): Value examples: Alumni, Bronfman, Chandler
  - room_number (varchar(7)): Value examples: 143, 547, 700
  - capacity (numeric(4,0)): Value examples: 10, 27, 92
Primary key: building, room_number

Table name: department
Columns:
  - dept_name (varchar(20)): Value examples: Accounting, Astronomy, Athletics
  - building (varchar(15)): Value examples: Chandler, Candlestick, Taylor
  - budget (numeric(12,2)): Value examples: 255041.46, 647610.55, 699140.86
Primary key: dept_name

Table name: course
Columns:
  - course_id (varchar(8)): Value examples: 101, 105, 123
  - title (varchar(50)): Value examples: C  Programming, The Music of Donovan, Electron Microscopy
  - dept_name (varchar(20)): Value examples: Mech. Eng., Comp. Sci., Statistics
  - credits (numeric(2,0)): Value examples: 4, 3
Primary key: course_id

Table name: instructor
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 16807
  - name (varchar(20)): Value examples: McKinnon, Pingr, Mird
  - dept_name (varchar(20)): Value examples: Cybernetics, Statistics, Marketing
  - salary (numeric(8,2)): Value examples: 94333.99, 59303.62, 119921.41
Primary key: ID

Table name: section
Columns:
  - course_id (varchar(8)): Value examples: 105, 137, 158
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2009, 2002, 2008
  - building (varchar(15)): Value examples: Chandler, Gates, Whitman
  - room_number (varchar(7)): Value examples: 804, 314, 434
  - time_slot_id (varchar(4)): Value examples: N, K, O
Primary key: course_id, sec_id, semester, year

Table name: teaches
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 19368
  - course_id (varchar(8)): Value examples: 747, 169, 445
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Spring, Fall
  - year (numeric(4,0)): Value examples: 2004, 2007, 2001
Primary key: ID, course_id, sec_id, semester, year

Table name: student
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - name (varchar(20)): Value examples: Schrefl, Rumat, Miliko
  - dept_name (varchar(20)): Value examples: History, Finance, Statistics
  - tot_cred (numeric(3,0)): Value examples: 4, 100, 116
Primary key: ID

Table name: takes
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - course_id (varchar(8)): Value examples: 239, 319, 362
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2006, 2003, 2005
  - grade (varchar(2)): Value examples: C-, B-, A-
Primary key: ID, course_id, sec_id, semester, year

Table name: advisor
Columns:
  - s_ID (varchar(5)): Value examples: 1000, 10033, 10076
  - i_ID (varchar(5)): Value examples: 19368, 95030, 22591
Primary key: s_ID

Table name: time_slot
Columns:
  - time_slot_id (varchar(4)): Value examples: A, B, C
  - day (varchar(1)): Value examples: F, M, W
  - start_hr (numeric(2)): Value examples: 8, 9, 11
  - start_min (numeric(2)): Value examples: 0, 30
  - end_hr (numeric(2)): Value examples: 8, 9, 11
  - end_min (numeric(2)): Value examples: 50, 45, 30
Primary key: time_slot_id, day, start_hr, start_min

Table name: prereq
Columns:
  - course_id (varchar(8)): Value examples: 133, 158, 169
  - prereq_id (varchar(8)): Value examples: 130, 169, 345
Primary key: course_id, prereq_id

Foreign keys:
  course.dept_name = department.dept_name
  instructor.dept_name = department.dept_name
  section.building = classroom.building
  section.room_number = classroom.room_number
  section.course_id = course.course_id
  teaches.ID = instructor.ID
  teaches.course_id = section.course_id
  teaches.sec_id = section.sec_id
  teaches.semester = section.semester
  teaches.year = section.year
  student.dept_name = department.dept_name
  takes.ID = student.ID
  takes.course_id = section.course_id
  takes.sec_id = section.sec_id
  takes.semester = section.semester
  takes.year = section.year
  advisor.s_ID = student.ID
  advisor.i_ID = instructor.ID
  prereq.prereq_id = course.course_id
  prereq.course_id = course.course_id
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select sum(department.budget) from department  where  department.dept_name = \"Marketing\"  or  department.dept_name = \"Finance\"
SQL: SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'

Medium example 2:
Q: "Find the name and building of the department with the highest budget."
Database: college_2

Table name: classroom
Columns:
  - building (varchar(15)): Value examples: Alumni, Bronfman, Chandler
  - room_number (varchar(7)): Value examples: 143, 547, 700
  - capacity (numeric(4,0)): Value examples: 10, 27, 92
Primary key: building, room_number

Table name: department
Columns:
  - dept_name (varchar(20)): Value examples: Accounting, Astronomy, Athletics
  - building (varchar(15)): Value examples: Chandler, Candlestick, Taylor
  - budget (numeric(12,2)): Value examples: 255041.46, 647610.55, 699140.86
Primary key: dept_name

Table name: course
Columns:
  - course_id (varchar(8)): Value examples: 101, 105, 123
  - title (varchar(50)): Value examples: C  Programming, The Music of Donovan, Electron Microscopy
  - dept_name (varchar(20)): Value examples: Mech. Eng., Comp. Sci., Statistics
  - credits (numeric(2,0)): Value examples: 4, 3
Primary key: course_id

Table name: instructor
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 16807
  - name (varchar(20)): Value examples: McKinnon, Pingr, Mird
  - dept_name (varchar(20)): Value examples: Cybernetics, Statistics, Marketing
  - salary (numeric(8,2)): Value examples: 94333.99, 59303.62, 119921.41
Primary key: ID

Table name: section
Columns:
  - course_id (varchar(8)): Value examples: 105, 137, 158
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2009, 2002, 2008
  - building (varchar(15)): Value examples: Chandler, Gates, Whitman
  - room_number (varchar(7)): Value examples: 804, 314, 434
  - time_slot_id (varchar(4)): Value examples: N, K, O
Primary key: course_id, sec_id, semester, year

Table name: teaches
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 19368
  - course_id (varchar(8)): Value examples: 747, 169, 445
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Spring, Fall
  - year (numeric(4,0)): Value examples: 2004, 2007, 2001
Primary key: ID, course_id, sec_id, semester, year

Table name: student
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - name (varchar(20)): Value examples: Schrefl, Rumat, Miliko
  - dept_name (varchar(20)): Value examples: History, Finance, Statistics
  - tot_cred (numeric(3,0)): Value examples: 4, 100, 116
Primary key: ID

Table name: takes
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - course_id (varchar(8)): Value examples: 239, 319, 362
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2006, 2003, 2005
  - grade (varchar(2)): Value examples: C-, B-, A-
Primary key: ID, course_id, sec_id, semester, year

Table name: advisor
Columns:
  - s_ID (varchar(5)): Value examples: 1000, 10033, 10076
  - i_ID (varchar(5)): Value examples: 19368, 95030, 22591
Primary key: s_ID

Table name: time_slot
Columns:
  - time_slot_id (varchar(4)): Value examples: A, B, C
  - day (varchar(1)): Value examples: F, M, W
  - start_hr (numeric(2)): Value examples: 8, 9, 11
  - start_min (numeric(2)): Value examples: 0, 30
  - end_hr (numeric(2)): Value examples: 8, 9, 11
  - end_min (numeric(2)): Value examples: 50, 45, 30
Primary key: time_slot_id, day, start_hr, start_min

Table name: prereq
Columns:
  - course_id (varchar(8)): Value examples: 133, 158, 169
  - prereq_id (varchar(8)): Value examples: 130, 169, 345
Primary key: course_id, prereq_id

Foreign keys:
  course.dept_name = department.dept_name
  instructor.dept_name = department.dept_name
  section.building = classroom.building
  section.room_number = classroom.room_number
  section.course_id = course.course_id
  teaches.ID = instructor.ID
  teaches.course_id = section.course_id
  teaches.sec_id = section.sec_id
  teaches.semester = section.semester
  teaches.year = section.year
  student.dept_name = department.dept_name
  takes.ID = student.ID
  takes.course_id = section.course_id
  takes.sec_id = section.sec_id
  takes.semester = section.semester
  takes.year = section.year
  advisor.s_ID = student.ID
  advisor.i_ID = instructor.ID
  prereq.prereq_id = course.course_id
  prereq.course_id = course.course_id
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select department.dept_name , department.building from department  order by department.budget desc limit 1
SQL: SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1

Hard example 1:
Q: "Find the title of courses that have two prerequisites?"
Database: college_2

Table name: classroom
Columns:
  - building (varchar(15)): Value examples: Alumni, Bronfman, Chandler
  - room_number (varchar(7)): Value examples: 143, 547, 700
  - capacity (numeric(4,0)): Value examples: 10, 27, 92
Primary key: building, room_number

Table name: department
Columns:
  - dept_name (varchar(20)): Value examples: Accounting, Astronomy, Athletics
  - building (varchar(15)): Value examples: Chandler, Candlestick, Taylor
  - budget (numeric(12,2)): Value examples: 255041.46, 647610.55, 699140.86
Primary key: dept_name

Table name: course
Columns:
  - course_id (varchar(8)): Value examples: 101, 105, 123
  - title (varchar(50)): Value examples: C  Programming, The Music of Donovan, Electron Microscopy
  - dept_name (varchar(20)): Value examples: Mech. Eng., Comp. Sci., Statistics
  - credits (numeric(2,0)): Value examples: 4, 3
Primary key: course_id

Table name: instructor
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 16807
  - name (varchar(20)): Value examples: McKinnon, Pingr, Mird
  - dept_name (varchar(20)): Value examples: Cybernetics, Statistics, Marketing
  - salary (numeric(8,2)): Value examples: 94333.99, 59303.62, 119921.41
Primary key: ID

Table name: section
Columns:
  - course_id (varchar(8)): Value examples: 105, 137, 158
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2009, 2002, 2008
  - building (varchar(15)): Value examples: Chandler, Gates, Whitman
  - room_number (varchar(7)): Value examples: 804, 314, 434
  - time_slot_id (varchar(4)): Value examples: N, K, O
Primary key: course_id, sec_id, semester, year

Table name: teaches
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 19368
  - course_id (varchar(8)): Value examples: 747, 169, 445
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Spring, Fall
  - year (numeric(4,0)): Value examples: 2004, 2007, 2001
Primary key: ID, course_id, sec_id, semester, year

Table name: student
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - name (varchar(20)): Value examples: Schrefl, Rumat, Miliko
  - dept_name (varchar(20)): Value examples: History, Finance, Statistics
  - tot_cred (numeric(3,0)): Value examples: 4, 100, 116
Primary key: ID

Table name: takes
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - course_id (varchar(8)): Value examples: 239, 319, 362
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2006, 2003, 2005
  - grade (varchar(2)): Value examples: C-, B-, A-
Primary key: ID, course_id, sec_id, semester, year

Table name: advisor
Columns:
  - s_ID (varchar(5)): Value examples: 1000, 10033, 10076
  - i_ID (varchar(5)): Value examples: 19368, 95030, 22591
Primary key: s_ID

Table name: time_slot
Columns:
  - time_slot_id (varchar(4)): Value examples: A, B, C
  - day (varchar(1)): Value examples: F, M, W
  - start_hr (numeric(2)): Value examples: 8, 9, 11
  - start_min (numeric(2)): Value examples: 0, 30
  - end_hr (numeric(2)): Value examples: 8, 9, 11
  - end_min (numeric(2)): Value examples: 50, 45, 30
Primary key: time_slot_id, day, start_hr, start_min

Table name: prereq
Columns:
  - course_id (varchar(8)): Value examples: 133, 158, 169
  - prereq_id (varchar(8)): Value examples: 130, 169, 345
Primary key: course_id, prereq_id

Foreign keys:
  course.dept_name = department.dept_name
  instructor.dept_name = department.dept_name
  section.building = classroom.building
  section.room_number = classroom.room_number
  section.course_id = course.course_id
  teaches.ID = instructor.ID
  teaches.course_id = section.course_id
  teaches.sec_id = section.sec_id
  teaches.semester = section.semester
  teaches.year = section.year
  student.dept_name = department.dept_name
  takes.ID = student.ID
  takes.course_id = section.course_id
  takes.sec_id = section.sec_id
  takes.semester = section.semester
  takes.year = section.year
  advisor.s_ID = student.ID
  advisor.i_ID = instructor.ID
  prereq.prereq_id = course.course_id
  prereq.course_id = course.course_id
A: Let's think step by step. "Find the title of courses that have two prerequisites?" can be solved by knowing the answer to the following sub-question "What are the titles for courses with two prerequisites?".
The SQL query for the sub-question "What are the titles for courses with two prerequisites?" is SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2
So, the answer to the question "Find the title of courses that have two prerequisites?" is =
Intermediate_representation: select course.title from course  where  count ( prereq.* )  = 2  group by prereq.course_id
SQL: SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2

Q: "Find the name and building of the department with the highest budget."
Database: college_2

Table name: classroom
Columns:
  - building (varchar(15)): Value examples: Alumni, Bronfman, Chandler
  - room_number (varchar(7)): Value examples: 143, 547, 700
  - capacity (numeric(4,0)): Value examples: 10, 27, 92
Primary key: building, room_number

Table name: department
Columns:
  - dept_name (varchar(20)): Value examples: Accounting, Astronomy, Athletics
  - building (varchar(15)): Value examples: Chandler, Candlestick, Taylor
  - budget (numeric(12,2)): Value examples: 255041.46, 647610.55, 699140.86
Primary key: dept_name

Table name: course
Columns:
  - course_id (varchar(8)): Value examples: 101, 105, 123
  - title (varchar(50)): Value examples: C  Programming, The Music of Donovan, Electron Microscopy
  - dept_name (varchar(20)): Value examples: Mech. Eng., Comp. Sci., Statistics
  - credits (numeric(2,0)): Value examples: 4, 3
Primary key: course_id

Table name: instructor
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 16807
  - name (varchar(20)): Value examples: McKinnon, Pingr, Mird
  - dept_name (varchar(20)): Value examples: Cybernetics, Statistics, Marketing
  - salary (numeric(8,2)): Value examples: 94333.99, 59303.62, 119921.41
Primary key: ID

Table name: section
Columns:
  - course_id (varchar(8)): Value examples: 105, 137, 158
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2009, 2002, 2008
  - building (varchar(15)): Value examples: Chandler, Gates, Whitman
  - room_number (varchar(7)): Value examples: 804, 314, 434
  - time_slot_id (varchar(4)): Value examples: N, K, O
Primary key: course_id, sec_id, semester, year

Table name: teaches
Columns:
  - ID (varchar(5)): Value examples: 14365, 15347, 19368
  - course_id (varchar(8)): Value examples: 747, 169, 445
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Spring, Fall
  - year (numeric(4,0)): Value examples: 2004, 2007, 2001
Primary key: ID, course_id, sec_id, semester, year

Table name: student
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - name (varchar(20)): Value examples: Schrefl, Rumat, Miliko
  - dept_name (varchar(20)): Value examples: History, Finance, Statistics
  - tot_cred (numeric(3,0)): Value examples: 4, 100, 116
Primary key: ID

Table name: takes
Columns:
  - ID (varchar(5)): Value examples: 1000, 10033, 10076
  - course_id (varchar(8)): Value examples: 239, 319, 362
  - sec_id (varchar(8)): Value examples: 1, 2, 3
  - semester (varchar(6)): Value examples: Fall, Spring
  - year (numeric(4,0)): Value examples: 2006, 2003, 2005
  - grade (varchar(2)): Value examples: C-, B-, A-
Primary key: ID, course_id, sec_id, semester, year

Table name: advisor
Columns:
  - s_ID (varchar(5)): Value examples: 1000, 10033, 10076
  - i_ID (varchar(5)): Value examples: 19368, 95030, 22591
Primary key: s_ID

Table name: time_slot
Columns:
  - time_slot_id (varchar(4)): Value examples: A, B, C
  - day (varchar(1)): Value examples: F, M, W
  - start_hr (numeric(2)): Value examples: 8, 9, 11
  - start_min (numeric(2)): Value examples: 0, 30
  - end_hr (numeric(2)): Value examples: 8, 9, 11
  - end_min (numeric(2)): Value examples: 50, 45, 30
Primary key: time_slot_id, day, start_hr, start_min

Table name: prereq
Columns:
  - course_id (varchar(8)): Value examples: 133, 158, 169
  - prereq_id (varchar(8)): Value examples: 130, 169, 345
Primary key: course_id, prereq_id

Foreign keys:
  course.dept_name = department.dept_name
  instructor.dept_name = department.dept_name
  section.building = classroom.building
  section.room_number = classroom.room_number
  section.course_id = course.course_id
  teaches.ID = instructor.ID
  teaches.course_id = section.course_id
  teaches.sec_id = section.sec_id
  teaches.semester = section.semester
  teaches.year = section.year
  student.dept_name = department.dept_name
  takes.ID = student.ID
  takes.course_id = section.course_id
  takes.sec_id = section.sec_id
  takes.semester = section.semester
  takes.year = section.year
  advisor.s_ID = student.ID
  advisor.i_ID = instructor.ID
  prereq.prereq_id = course.course_id
  prereq.course_id = course.course_id
A: Let's think step by step. "Find the name and building of the department with the highest budget." can be solved by knowing the answer to the following sub-question "What is the department name and corresponding building for the department with the greatest budget?".
The SQL query for the sub-question "What is the department name and corresponding building for the department with the greatest budget?" is SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1
So, the answer to the question "Find the name and building of the department with the highest budget." is =
Intermediate_representation: select department.dept_name , department.building from department  order by department.budget desc limit 1
SQL: SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1
"""