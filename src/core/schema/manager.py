from typing import Dict, Optional, Union
from pathlib import Path
from threading import Lock
from .schema import DatabaseSchema
from .extractor import SchemaExtractor
from ..config import Config

class SchemaManager:
    """Class for managing database schema"""
    _instance = None
    _lock = Lock()
    _schema_cache: Dict[str, DatabaseSchema] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_schema(self, db_id: str, db_path: str) -> DatabaseSchema:
        """
        Get database schema, prioritize to get from cache
        :param db_id: database ID
        :param db_path: database file path
        :return: database schema
        """
        if db_id not in self._schema_cache:
            with self._lock:
                if db_id not in self._schema_cache:
                    self._schema_cache[db_id] = SchemaExtractor.extract_from_db(db_path, db_id)
        return self._schema_cache[db_id]
    
    def get_formatted_enriched_schema(self, db_id: str, source: str) -> str:
        """
        Get formatted schema string by database ID and source
        :param db_id: database ID
        :param source: database source
        :return: formatted schema string
        """
        # Build database path
        db_folder = f"{source}_{db_id}"
        db_file = f"{db_id}.sqlite"
        db_path = Config().database_dir / db_folder / db_file
        
        try:
            schema = self.get_schema(db_id, str(db_path))
            schema_dict = schema.to_dict()
            
            # Format schema information
            result = []
            
            # Add database name
            result.append(f"Database: {schema_dict['database']}\n")
            
            # Format each table information
            for table in schema_dict['tables']:
                result.append(f"Table name: {table['table']}")
                
                # Add column information
                result.append("Columns:")
                for col_name, col_info in table['columns'].items():
                    col_desc = []
                    if col_info['expanded_name']:
                        col_desc.append(f"Meaning: {col_info['expanded_name']}")
                    if col_info['description']:
                        col_desc.append(f"Description: {col_info['description']}")
                    if col_info['data_format']:
                        col_desc.append(f"Format: {col_info['data_format']}")
                    if col_info['value_description']:
                        col_desc.append(f"Value description: {col_info['value_description']}")
                    if col_info['value_examples']:
                        col_desc.append(f"Value examples: {', '.join(col_info['value_examples'])}")
                        
                    if col_desc:
                        result.append(f"  - {col_name} ({col_info['type']}): {' | '.join(col_desc)}")
                    else:
                        result.append(f"  - {col_name} ({col_info['type']})")
                        
                # Add primary key information
                if table['primary_keys']:
                    result.append(f"Primary key: {', '.join(table['primary_keys'])}")
                    
                result.append("")  # Add empty line for separation
                
            # Add foreign key relationship
            if schema_dict.get('foreign_keys'):
                result.append("Foreign keys:")
                for fk in schema_dict['foreign_keys']:
                    result.append(
                        f"  {fk['table'][0]}.{fk['column'][0]} = "
                        f"{fk['table'][1]}.{fk['column'][1]}"
                    )
                    
            return "\n".join(result)
            
        except Exception as e:
            raise ValueError(f"Access schema failure: {str(e)}")
    
    def get_db_schema_by_id_source(self, db_id: str, source: str) -> Dict:
        """
        Get database schema with additional information by database ID and source
        
        :param db_id: database ID
        :param source: database source
        :return: schema dictionary with additional information
        """
        # Build database path
        db_folder = f"{source}_{db_id}"  # e.g. spider_dev_academic
        db_file = f"{db_id}.sqlite"      # e.g. academic.sqlite
        db_path = str(Config().database_dir / db_folder / db_file)
        
        try:
            # Get schema and convert to dictionary format
            schema = self.get_schema(db_id, db_path)
            return schema.to_dict()
        except Exception as e:
            raise ValueError(f"Access schema failure: {str(e)}")
    
    def clear_cache(self):
        """Clear cache"""
        with self._lock:
            self._schema_cache.clear() 
    
    def format_enriched_db_schema(self, schema: Dict) -> str:
        """
        Format enriched database schema information for SQL generation module prompt
        :param schema: database schema dictionary
        :return: formatted schema string
        """
        result = []
        
        # Add database name
        result.append(f"Database: {schema['database']}\n")
        
        # Format each table information
        for table in schema['tables']:
            result.append(f"Table name: {table['table']}")
            
            # Add column information
            result.append("Columns:")
            for col_name, col_info in table['columns'].items():
                col_desc = []
                if col_info['expanded_name']:
                    col_desc.append(f"Meaning: {col_info['expanded_name']}")
                if col_info['description']:
                    col_desc.append(f"Description: {col_info['description']}")
                if col_info['data_format']:
                    col_desc.append(f"Format: {col_info['data_format']}")
                if col_info['value_description']:
                    col_desc.append(f"Value description: {col_info['value_description']}")
                if col_info['value_examples']:
                    col_desc.append(f"Value examples: {', '.join(col_info['value_examples'])}")
                    
                if col_desc:
                    result.append(f"  - {col_name} ({col_info['type']}): {' | '.join(col_desc)}")
                else:
                    result.append(f"  - {col_name} ({col_info['type']})")
                    
            # Add primary key information
            if table['primary_keys']:
                result.append(f"Primary key: {', '.join(table['primary_keys'])}")
                
            result.append("")  # Add empty line for separation
            
        # Add foreign key relationship
        if schema.get('foreign_keys'):
            result.append("Foreign keys:")
            for fk in schema['foreign_keys']:
                result.append(
                    f"  {fk['table'][0]}.{fk['column'][0]} = "
                    f"{fk['table'][1]}.{fk['column'][1]}"
                )
                
        return "\n".join(result)
    
    def format_linked_schema(self, linked_schema: Dict) -> str:
        """
        Format linked schema for SQL generation module prompt
        :param linked_schema: schema after Schema Linking
        :return: formatted schema string
        """
        result = []
        
        # Format each table information
        for table in linked_schema["tables"]:
            table_name = table["table"]
            columns = table.get("columns", [])
            columns_info = table.get("columns_info", {})
            
            # Add table information
            result.append(f"Table: {table_name}")
            
            # Add column information (with additional information)
            if columns:
                result.append("Columns:")
                for col in columns:
                    col_info = columns_info.get(col, {})
                    col_desc = []
                    
                    # Add column type
                    col_type = col_info.get("type", "")
                    if col_type:
                        col_desc.append(f"Type: {col_type}")
                    
                    # Add other additional information
                    if col_info.get("expanded_name"):
                        col_desc.append(f"Meaning: {col_info['expanded_name']}")
                    if col_info.get("description"):
                        col_desc.append(f"Description: {col_info['description']}")
                    if col_info.get("data_format"):
                        col_desc.append(f"Format: {col_info['data_format']}")
                    if col_info.get("value_description"):
                        col_desc.append(f"Value description: {col_info['value_description']}")
                    if col_info.get("value_examples"):
                        col_desc.append(f"Value examples: {', '.join(col_info['value_examples'])}")
                    
                    if col_desc:
                        result.append(f"  - {col}: {' | '.join(col_desc)}")
                    else:
                        result.append(f"  - {col}")
            
            # Add primary key information
            if "primary_keys" in table:
                result.append(f"Primary key: {', '.join(table['primary_keys'])}")
            
            # Add foreign key information
            if "foreign_keys" in table:
                result.append("Foreign keys:")
                for fk in table["foreign_keys"]:
                    result.append(
                        f"  - {fk['column']} -> "
                        f"{fk['referenced_table']}.{fk['referenced_column']}"
                    )
            
            result.append("")  # Add empty line for separation
            
        return "\n".join(result) 