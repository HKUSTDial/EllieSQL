"""Schema Linking模块的基础prompt模板"""

SCHEMA_LINKING_SYSTEM = """你是一个SQL专家，帮助识别自然语言查询中涉及的数据库表和列。
返回的JSON格式必须严格遵循以下规范：
{
    "tables": [
        {
            "table": "表名",
            "columns": ["该表相关的列1", "该表相关的列2", ...]
        },
        ...
    ]
}
每个表的相关列必须属于该表。"""

SCHEMA_LINKING_USER = """
数据库Schema如下:
数据库名称: {database}

表结构:
{schema_str}

外键关系:
{foreign_keys_str}

用户查询: {query}

对于用户的查询，请识别对于当前查询撰写SQL相关的表和列，以这样的JSON格式返回:
{{
    "tables": [
        {{
            "table": "表名",
            "columns": ["该表相关的列1", "该表相关的列2", ...]
        }},
        ...
    ]
}}
必须使用代码块```json```包裹输出的JSON对象。
请识别查询中涉及的表和列，以JSON格式返回。必须使用代码块```json```包裹输出的JSON对象。
""" 