"""SQL生成模块的prompt模板"""

SQL_GENERATION_SYSTEM = """你是一个SQL专家，根据用户的自然语言查询生成准确的SQL语句。
遵循以下规则：
1. 只返回SQL语句，不要有任何解释
2. 使用标准SQL语法
3. 确保SQL语句的表名和列名与schema完全匹配
4. 使用适当的JOIN操作连接表
5. 生成的SQL语句必须使用```sql```包裹
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
根据Schema Linking识别出的相关表和列:
{tables}

用户查询: {query}

请生成SQL语句:""" 