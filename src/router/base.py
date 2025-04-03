from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ..modules.sql_generation.base import SQLGeneratorBase

class RouterBase(SQLGeneratorBase):
    """The base class of Router (inherit SQLGeneratorBase class for easy implementation)"""
    
    def __init__(self, name: str = "BaseRouter"):
        super().__init__(name)
        self.generators: Dict[str, SQLGeneratorBase] = {}
        
    def register_generator(self, name: str, generator: SQLGeneratorBase):
        """Register the SQL generator"""
        self.generators[name] = generator
        # Pass the necessary attributes to the generator
        generator.set_previous_module(self._previous_module)
        if hasattr(self, 'data_file'):
            generator.set_data_file(self.data_file)
        if hasattr(self, 'logger'):
            generator.logger = self.logger
        
    def set_previous_module(self, module):
        """Set the previous module, and update the previous module of all generators"""
        super().set_previous_module(module)
        for generator in self.generators.values():
            generator.set_previous_module(module)
            
    def set_data_file(self, data_file: str):
        """Set the data file path, and update the data file path of all generators"""
        super().set_data_file(data_file)
        for generator in self.generators.values():
            generator.set_data_file(data_file)
            
    @abstractmethod
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """
        Select the appropriate SQL generator based on the input information
        
        :param query: user query
        :param schema_linking_output: the output of Schema Linking
        :param query_id: query ID
            
        :return: the name of the selected generator
        """
        pass
        
    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str = None, module_name: Optional[str] = None) -> str:
        """
        Select the appropriate generator through routing and generate SQL (implementation of the abstract method generate_sql of SQLGeneratorBase)
        
        :param query: user query
        :param schema_linking_output: the output of Schema Linking
        :param query_id: query ID
        :param module_name: module name
            
        :return: the generated SQL
        """
        # Get the routing result
        selected_generator_name = await self.route(query, schema_linking_output, query_id)
        
        if selected_generator_name not in self.generators:
            raise ValueError(f"Generator {selected_generator_name} not found")
            
        # Use the selected generator to generate SQL
        selected_generator = self.generators[selected_generator_name]
        raw_output = await selected_generator.generate_sql(query, schema_linking_output, query_id, module_name)
        extracted_sql = self.extractor.extract_sql(raw_output)
        
        # Save the intermediate results
        self.save_intermediate(
            input_data={
                "query": query,
                "linked_schema": schema_linking_output.get("linked_schema", {}),
            },
            output_data={
                "raw_output": raw_output,
                "extracted_sql": extracted_sql,
                "selected_generator": selected_generator_name,
            },
            model_info={
                "model": "none",  # router itself does not consume tokens
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            query_id=query_id,
            module_name=self.name if module_name is None else module_name
        )
        
        # Record the routing information
        self.log_io(
            input_data={
                "query": query,
                "schema_linking_output": schema_linking_output
            },
            output_data={
                "selected_generator": selected_generator_name,
                "raw_output": raw_output,
                "extracted_sql": extracted_sql
            }
        )
        
        return raw_output 