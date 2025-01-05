"""Schema Linking模块的统一prompt模板"""

BASE_SCHEMA_SYSTEM = """你是一个SQL专家，帮助识别自然语言查询中涉及的数据库表和列。
分析用户查询和数据库schema，识别相关的表和列。

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

BASE_SCHEMA_USER = """
数据库Schema信息如下:

{schema_str}

用户查询: {query}

请识别查询中涉及的表和列，以JSON格式返回。必须使用代码块```json```包裹输出的JSON对象。
"""

ENHANCED_SCHEMA_SYSTEM = BASE_SCHEMA_SYSTEM + """
注意每个表和列都有详细的描述信息和示例值，请利用这些信息提高匹配的准确性。""" 