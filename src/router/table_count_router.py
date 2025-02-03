from typing import Dict, Optional
from .base import RouterBase
from ..core.utils import load_json
from ..pipeline_factory import PipelineLevel

class TableCountRouter(RouterBase):
    """基于表数量的路由器"""
    
    def __init__(self, name: str = "TableCountRouter"):
        super().__init__(name)
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """
        根据schema linking结果中的表数量选择合适的SQL生成器: 
        1. 表数量小于2, 使用基础生成器
        2. 表数量在2到4之间, 使用中级生成器
        3. 表数量大于等于4, 使用高级生成器
        
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
        
        # 1. 表数量小于2，使用基础生成器
        if table_count < 2:
            # print('basic')
            return PipelineLevel.BASIC.value
        # 2. 表数量在2到4之间，使用中级生成器
        elif table_count < 4:
            # print('intermediate')
            return PipelineLevel.INTERMEDIATE.value
        # 3. 表数量大于等于4，使用高级生成器
        else:
            # print('advanced')
            return PipelineLevel.ADVANCED.value

    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str = None, module_name: Optional[str] = None) -> str:
        """
        通过路由选择合适的生成器并生成SQL (对RouterBase的generate_sql方法的重写)
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
            query_id: 查询ID
            module_name: 模块名称
            
        Returns:
            str: 生成的SQL
        """
        # 获取路由结果
        selected_generator_name = await self.route(query, schema_linking_output, query_id)
        
        if selected_generator_name not in self.generators:
            raise ValueError(f"Generator {selected_generator_name} not found")
            
        # 使用选定的生成器生成SQL
        selected_generator = self.generators[selected_generator_name]
        generator_result = await selected_generator.generate_sql(query, schema_linking_output, query_id, module_name)
        
        # 从生成器结果中提取SQL
        raw_output = generator_result
        extracted_sql = self.extractor.extract_sql(raw_output)
        
        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query,
                "linked_schema": schema_linking_output.get("linked_schema", {}),
                # "selected_generator": selected_generator_name
            },
            output_data={
                "raw_output": raw_output,
                "extracted_sql": extracted_sql,
                "selected_generator": selected_generator_name
            },
            model_info={
                "model": getattr(selected_generator, "model", "unknown"),
                "input_tokens": getattr(selected_generator, "last_input_tokens", 0),
                "output_tokens": getattr(selected_generator, "last_output_tokens", 0),
                "total_tokens": getattr(selected_generator, "last_total_tokens", 0)
            },
            query_id=query_id,
            module_name=self.name if module_name is None else module_name
        )
        
        # 记录路由信息
        self.log_io(
            input_data={
                "query": query,
                "schema_linking_output": schema_linking_output,
                # "selected_generator": selected_generator_name
            },
            output_data={
                "selected_generator": selected_generator_name,
                "raw_output": raw_output,
                "extracted_sql": extracted_sql
            }
        )
        
        return raw_output 