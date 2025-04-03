from typing import Dict
from .base import SchemaLinkerBase
from ...core.schema.manager import SchemaManager

class SkipSchemaLinker(SchemaLinkerBase):
    """Implementation of Schema Linking that skips the process and returns the *enhanced* database schema"""
    
    def __init__(self, max_retries: int = 1):
        super().__init__("SkipSchemaLinker", max_retries)
        self.schema_manager = SchemaManager()
        
    async def link_schema(self, query: str, database_schema: Dict, query_id: str = None) -> str:
        """
        Skip schema linking and return the *enhanced* database schema directly
        
        :param query: The user query
        :param database_schema: The database schema
        :param query_id: The query ID
            
        :return: str: The formatted schema string
        """
        try:
            # Construct a linked_schema containing all tables and columns
            linked_schema = {
                "tables": [
                    {
                        "table": table["table"],
                        "columns": list(table["columns"].keys()),
                        "columns_info": table["columns"]
                    }
                    for table in database_schema["tables"]
                ]
            }
            
            # Save the linked schema result
            source = database_schema.get("source", "unknown")
            self.save_linked_schema_result(
                query_id=query_id,
                source=source,
                linked_schema={
                    "database": database_schema.get("database", ""),
                    "tables": linked_schema.get("tables", [])
                }
            )
            
            # Format the linked schema
            formatted_linked_schema = self.schema_manager.format_linked_schema(linked_schema)
            # print("****formatted_linked_schema*****\n", formatted_linked_schema, "\n*********")
            # Save intermediate results
            self.save_intermediate(
                input_data={
                    "query": query,
                    # "database_schema": database_schema
                },
                output_data={
                    # "linked_schema": linked_schema,
                    "formatted_linked_schema": formatted_linked_schema
                },
                model_info={
                    "model": "none",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                },
                query_id=query_id
            )

            # To maintain consistency with the interface, use a fixed raw_output content for placeholder
            raw_output = """```json {"tables": [{"table": "<skip schema linking>", "columns": ["<skip schema linking>"]}]}```"""

            self.log_io(
                {
                    "query": query, 
                    "database_schema": database_schema, 
                    "formatted_schema": formatted_linked_schema
                }, 
                raw_output
            )
            return raw_output
            
        except Exception as e:
            self.logger.error(f"Skip schema linking failed: {str(e)}")
            raise 