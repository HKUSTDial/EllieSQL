from typing import Dict
from .base import RouterBase
from ..pipeline_factory import PipelineLevel

class TableCountRouter(RouterBase):
    """Router based on the number of tables in the schema linking result"""
    
    def __init__(self, name: str = "TableCountRouter"):
        super().__init__(name)
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """
        Select the appropriate SQL generator based on the number of tables in the schema linking result:
        1. If the number of tables is less than or equal to 2, use the basic generator
        2. If the number of tables is equal to 3, use the intermediate generator
        3. If the number of tables is greater than 3, use the advanced generator
        
        :param query: user query
        :param schema_linking_output: Schema Linking output
        :param query_id: query ID
            
        :return: the name of the selected generator (corresponding to PipelineLevel)
        """
        linked_schema = schema_linking_output.get("linked_schema", {})
        tables = linked_schema.get("tables", [])
        table_count = len(tables)
        
        assert table_count >= 1 and type(table_count) == int, f"Schema Linking result table count is abnormal! Question ID: {query_id}, table count: {table_count}"
        if table_count < 2:
            return PipelineLevel.BASIC.value
        elif table_count >= 2 and table_count <= 3:
            return PipelineLevel.INTERMEDIATE.value
        elif table_count > 3:
            return PipelineLevel.ADVANCED.value
        else:
            raise ValueError(f"Schema Linking result table count is abnormal! Question ID: {query_id}, table count: {table_count}")