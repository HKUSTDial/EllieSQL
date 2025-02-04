from typing import Dict
from .base import RouterBase
from ..pipeline_factory import PipelineLevel

class TableCountRouter(RouterBase):
    """基于Schema Linking结果中涉及的表数量进行路由的路由器"""
    
    def __init__(self, name: str = "TableCountRouter"):
        super().__init__(name)
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """
        根据schema linking结果中的表数量选择合适的SQL生成器: 
        1. 表数量小于等于2, 使用基础生成器
        2. 表数量等于3, 使用中级生成器
        3. 表数量大于3, 使用高级生成器
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
            query_id: 查询ID
            
        Returns:
            str: 选择的生成器名称 (对应 PipelineLevel)
        """
        linked_schema = schema_linking_output.get("linked_schema", {})
        tables = linked_schema.get("tables", [])
        table_count = len(tables)
        
        assert table_count >= 1 and type(table_count) == int, f"Schema Linking结果中涉及的表数量异常! Question ID: {query_id}, 表数量: {table_count}"
        if table_count < 2:
            return PipelineLevel.BASIC.value
        elif table_count >= 2 and table_count <= 3:
            return PipelineLevel.INTERMEDIATE.value
        elif table_count > 3:
            return PipelineLevel.ADVANCED.value
        else:
            raise ValueError(f"Schema Linking结果中涉及的表数量异常! Question ID: {query_id}, 表数量: {table_count}")