from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ..modules.sql_generation.base import SQLGeneratorBase

class RouterBase(SQLGeneratorBase):
    """Router的基类"""
    
    def __init__(self, name: str = "BaseRouter"):
        super().__init__(name)
        self.generators: Dict[str, SQLGeneratorBase] = {}
        
    def register_generator(self, name: str, generator: SQLGeneratorBase):
        """注册SQL生成器"""
        self.generators[name] = generator
        # 传递必要的属性给生成器
        generator.set_previous_module(self._previous_module)
        if hasattr(self, 'data_file'):
            generator.set_data_file(self.data_file)
        if hasattr(self, 'logger'):
            generator.logger = self.logger
        
    def set_previous_module(self, module):
        """设置前置模块，同时更新所有生成器的前置模块"""
        super().set_previous_module(module)
        for generator in self.generators.values():
            generator.set_previous_module(module)
            
    def set_data_file(self, data_file: str):
        """设置数据文件路径，同时更新所有生成器的数据文件路径"""
        super().set_data_file(data_file)
        for generator in self.generators.values():
            generator.set_data_file(data_file)
            
    @abstractmethod
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """
        根据输入信息选择合适的SQL生成器
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
            query_id: 查询ID
            
        Returns:
            str: 生成的SQL
        """
        pass
        
    @abstractmethod
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
        pass 