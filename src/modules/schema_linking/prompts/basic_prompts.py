"""Schema Linking模块的基础prompt模板"""

SCHEMA_LINKING_SYSTEM = """你是一个SQL专家，帮助识别自然语言查询中涉及的数据库表和列。"""

SCHEMA_LINKING_USER = """
数据库Schema如下:
{schema_str}

用户查询: {query}

请识别查询中涉及的表和列，以JSON格式返回:
{{
    "tables": ["涉及的表名1", "表名2", ...],
    "columns": ["涉及的列名1", "列名2", ...]
}}
只返回JSON，不要其他解释。
""" 