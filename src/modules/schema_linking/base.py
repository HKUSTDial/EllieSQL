from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from ..base import ModuleBase
import json
import os

class SchemaLinkerBase(ModuleBase):
    """Base class for the Schema Linking module"""
    
    def __init__(self, name: str, max_retries: int = 3):
        super().__init__(name)
        self.max_retries = max_retries  # Retry threshold
        
    def enrich_schema_info(self, linked_schema: Dict, database_schema: Dict) -> Dict:
        """
        Enrich the linked schema with primary and foreign key information
        
        :param linked_schema: The original output of Schema Linking
        :param database_schema: The complete database schema
            
        :return: The schema with primary and foreign key information
        """
        # Create a mapping from table names to table information
        db_tables = {table["table"]: table for table in database_schema["tables"]}
        
        # Enrich each linked table with information
        for table in linked_schema["tables"]:
            table_name = table["table"]
            if table_name in db_tables:
                # Enrich primary key information
                if "primary_keys" in db_tables[table_name]:
                    table["primary_keys"] = db_tables[table_name]["primary_keys"]
                
        # Enrich foreign key information
        if "foreign_keys" in database_schema:
            for fk in database_schema["foreign_keys"]:
                # Check if the table related to the foreign key is in the linked schema
                src_table = fk["table"][0]
                dst_table = fk["table"][1]
                linked_tables = [t["table"] for t in linked_schema["tables"]]
                
                if src_table in linked_tables and dst_table in linked_tables:
                    # Find the source table
                    for table in linked_schema["tables"]:
                        if table["table"] == src_table:
                            if "foreign_keys" not in table:
                                table["foreign_keys"] = []
                            # Add foreign key information
                            table["foreign_keys"].append({
                                "column": fk["column"][0],
                                "referenced_table": dst_table,
                                "referenced_column": fk["column"][1]
                            })
                            
        return linked_schema
    
    @abstractmethod
    async def link_schema(self, 
                       query: str, 
                       database_schema: Dict) -> Dict:
        """
        Link the natural language query with the database schema
        
        :param query: The user's natural language query
        :param database_schema: The database schema information
            
        :return: The linked schema information
        """
        pass
    
    async def link_schema_with_retry(self, query: str, database_schema: Dict, query_id: str = None) -> str:
        """
        Schema linking with retry mechanism, returning the full database schema after reaching the retry threshold
        
        :param query: The user's query
        :param database_schema: The database schema
        :param query_id: The query ID
            
        :return: The result of Schema Linking (with primary and foreign key information) or the full database schema in JSON string format
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw_schema_output = await self.link_schema(query, database_schema, query_id)
                if raw_schema_output is not None:
                    # Extract the schema from the original output and enrich it with primary and foreign key information
                    extracted_linked_schema = self.extractor.extract_schema_json(raw_schema_output)
                    enriched_linked_schema = self.enrich_schema_info(extracted_linked_schema, database_schema)
                    if extracted_linked_schema is not None:
                        return enriched_linked_schema
            except Exception as e:
                last_error = e
                self.logger.warning(f"Schema linking failed on the {attempt + 1}/{self.max_retries} attempt: {str(e)}")
                continue
        
        # Reaching the retry threshold, returning the full database schema, ensuring the format is consistent with the normal return
        self.logger.error(f"Schema linking failed after {self.max_retries} attempts, returning the full database schema. Last error: {str(last_error)}. Question ID: {query_id} The program continues to execute...")
        
        # Construct a schema containing all tables and columns, and enrich it with primary and foreign key information
        full_schema = {
            "tables": [
                {
                    "table": table["table"],
                    "columns": list(table["columns"].keys()),
                    "columns_info": table["columns"],
                    "primary_keys": table.get("primary_keys", [])
                }
                for table in database_schema["tables"]
            ]
        }
        
        # Add foreign key information
        if "foreign_keys" in database_schema:
            for table in full_schema["tables"]:
                table_name = table["table"]
                table["foreign_keys"] = []
                for fk in database_schema["foreign_keys"]:
                    if fk["table"][0] == table_name:
                        table["foreign_keys"].append({
                            "column": fk["column"][0],
                            "referenced_table": fk["table"][1],
                            "referenced_column": fk["column"][1]
                        })
        
        # Save intermediate results
        self.save_intermediate(
            input_data={
                "query": query,
                # "database_schema": database_schema
            },
            output_data={
                "raw_output": "Schema linking failed, using full database schema",
                "extracted_linked_schema": full_schema,
                "formatted_linked_schema": self.schema_manager.format_linked_schema(full_schema)
            },
            model_info={
                "model": "none",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            query_id=query_id
        )

        # Even if Schema Linking fails, return the full database schema and save the result
        source = database_schema.get("source", "unknown")
        self.save_linked_schema_result(
            query_id=query_id,
            source=source,
            linked_schema={
                "database": database_schema.get("database", ""),
                "tables": full_schema.get("tables", [])
            }
        )
        return full_schema
    
    def _format_basic_schema(self, schema: Dict) -> str:
        """Basic schema formatting method"""
        result = []
        
        # Add the database name
        result.append(f"Database: {schema['database']}\n")
        
        # Format the table structure
        for table in schema['tables']:
            result.append(f"Table name: {table['table']}")
            result.append(f"Columns: {', '.join(table['columns'])}")
            if table['primary_keys']:
                result.append(f"Primary keys: {', '.join(table['primary_keys'])}")
            result.append("")

        # Format the foreign key relationship
        if schema.get('foreign_keys'):
            result.append("Foreign keys:")
            for fk in schema['foreign_keys']:
                result.append(
                    f"  {fk['table'][0]}.{fk['column'][0]} = "
                    f"{fk['table'][1]}.{fk['column'][1]}"
                )
                
        return "\n".join(result) 
    
    def save_linked_schema_result(self, query_id: str, source: str, linked_schema: Dict) -> None:
        """
        Save the linked schema result to a separate JSONL file.
        
        :param query_id: Query ID
        :param source: Data source (e.g., 'bird_dev')
        :param linked_schema: The linked schema with primary/foreign keys
        """
        # Get the pipeline directory from the intermediate result handler
        pipeline_dir = self.intermediate.pipeline_dir
        
        # Define the output file path
        output_file = os.path.join(pipeline_dir, "linked_schema_results.jsonl")
        
        # Format the result in the specified structure
        result = {
            "query_id": query_id,
            "source": source,
            "database": linked_schema.get("database", ""),
            "tables": [
                {
                    "table": table["table"],
                    "columns": table["columns"]
                }
                for table in linked_schema.get("tables", [])
            ]
        }
        
        # Save to JSONL file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n") 