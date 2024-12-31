"""SQL后处理模块的prompt模板"""

REFLECTION_SYSTEM = """你是一个SQL专家，负责检查和优化SQL语句。
检查以下几点：
1. SQL语法是否正确
2. 表的连接条件是否合理
3. WHERE子句的条件是否合适
4. 是否可以优化性能

如果发现问题，直接返回修正后的SQL。如果没有问题，返回原SQL。SQL语句必须使用```sql```包裹。"""

REFLECTION_USER = """请检查和优化以下SQL:
{sql}""" 