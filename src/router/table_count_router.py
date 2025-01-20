from typing import Dict, Optional
from .base import RouterBase

class TableCountRouter(RouterBase):
    """基于表数量的路由器"""
    
    def __init__(self, 
                single_table_generator: str = "vanilla",
                multi_table_generator: str = "enhanced"):
        super().__init__("TableCountRouter")
        self.single_table_generator = single_table_generator
        self.multi_table_generator = multi_table_generator
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """
        根据linked schema中的表数量选择SQL生成器
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
            query_id: 查询ID
            
        Returns:
            str: 生成的SQL
        """
        # 获取linked schema中的表数量
        linked_schema = schema_linking_output.get("linked_schema", {})
        tables = linked_schema.get("tables", [])
        table_count = len(tables)
        
        # 根据表数量选择生成器
        if table_count <= 2:
            generator_name = self.single_table_generator
        else:
            generator_name = self.multi_table_generator
            
        if generator_name not in self.generators:
            raise ValueError(f"Generator {generator_name} not registered")
            
        # 使用选中的生成器生成SQL
        generator = self.generators[generator_name]
        return await generator.generate_sql_with_retry(query, schema_linking_output, query_id)

    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str = None, module_name: Optional[str] = None) -> str:
        """
        通过路由选择合适的生成器并生成SQL
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
            query_id: 查询ID
            module_name: 模块名称
            
        Returns:
            str: 生成的SQL
        """
        # 获取路由后的SQL
        raw_output = await self.route(query, schema_linking_output, query_id)
        
        # 提取SQL
        extracted_sql = self.extractor.extract_sql(raw_output)
        
        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query,
                "linked_schema": schema_linking_output.get("linked_schema", {})
            },
            output_data={
                "raw_output": raw_output,
                "extracted_sql": extracted_sql,
                "selected_generator": self.generators[
                    self.single_table_generator if len(schema_linking_output.get("linked_schema", {}).get("tables", [])) <= 1 
                    else self.multi_table_generator
                ].name
            },
            model_info={
                "model": "router",
                "input_tokens": 0,  # Router本身不消耗token
                "output_tokens": 0,
                "total_tokens": 0
            },
            query_id=query_id,
            module_name=self.name if module_name is None else module_name
        )
        
        # 记录路由信息
        self.log_io(
            input_data={
                "query": query,
                "schema_linking_output": schema_linking_output
            },
            output_data={
                "selected_generator": self.generators[
                    self.single_table_generator if len(schema_linking_output.get("linked_schema", {}).get("tables", [])) <= 1 
                    else self.multi_table_generator
                ].name,
                "raw_output": raw_output
            }
        )
        
        return raw_output 