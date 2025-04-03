from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class ColumnSchema:
    """Column schema definition"""
    name: str
    type: str
    description: str = ""
    expanded_name: str = ""  # Extended column name (more understandable name)
    value_description: str = ""  # Value description
    data_format: str = ""  # Data format
    value_examples: List[str] = field(default_factory=list)
    is_primary_key: bool = False
    foreign_keys: List[Tuple[str, str]] = field(default_factory=list)  # [(target_table, target_column)]
    referenced_by: List[Tuple[str, str]] = field(default_factory=list)  # Referenced by which tables
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "expanded_name": self.expanded_name,
            "value_description": self.value_description,
            "data_format": self.data_format,
            "value_examples": self.value_examples,
            "is_primary_key": self.is_primary_key,
            "foreign_keys": self.foreign_keys,
            "referenced_by": self.referenced_by
        }

@dataclass
class TableSchema:
    """Table schema definition"""
    name: str
    columns: Dict[str, ColumnSchema]
    primary_keys: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "table": self.name,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
            "primary_keys": self.primary_keys
        }

@dataclass
class DatabaseSchema:
    """Database schema definition"""
    database: str
    tables: Dict[str, TableSchema]
    foreign_keys: List[Dict[str, List[str]]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "database": self.database,
            "tables": [table.to_dict() for table in self.tables.values()],
            "foreign_keys": self.foreign_keys,
            "foreign_key_count": len(self.foreign_keys),
            "table_count": len(self.tables),
            "total_column_count": sum(len(table.columns) for table in self.tables.values())
        }
    
    def get_table_ddl(self, 
                     add_expanded_name: bool = True,
                     add_description: bool = True,
                     add_value_description: bool = True,
                     add_examples: bool = True) -> str:
        """
        Generate DDL statements for the database

        :param add_expanded_name: whether to add extended column name
        :param add_description: whether to add column description
        :param add_value_description: whether to add value description
        :param add_examples: whether to add example values
        :return: DDL statements
        """
        ddl_statements = []
        for table in self.tables.values():
            ddl = f"CREATE TABLE `{table.name}` (\n"
            
            # Add column definitions
            for col_name, column in table.columns.items():
                col_def = f"\t`{col_name}` {column.type}"
                
                # Add comments
                comments = []
                if add_expanded_name and column.expanded_name:
                    comments.append(f"Column Meaning: {column.expanded_name}")
                if add_description and column.description:
                    comments.append(f"Description: {column.description}")
                if add_value_description and column.value_description:
                    comments.append(f"Values: {column.value_description}")
                if add_examples and column.value_examples:
                    comments.append(f"Examples: {', '.join(column.value_examples)}")
                
                if comments:
                    col_def += f" -- {' | '.join(comments)}"
                col_def += ",\n"
                ddl += col_def
            
            # Add primary key constraints
            if table.primary_keys:
                ddl += f"\tPRIMARY KEY ({', '.join(f'`{pk}`' for pk in table.primary_keys)}),\n"
            
            # Add foreign key constraints
            for col_name, column in table.columns.items():
                for fk_table, fk_column in column.foreign_keys:
                    ddl += f"\tFOREIGN KEY (`{col_name}`) REFERENCES `{fk_table}`(`{fk_column}`),\n"
            
            # Remove the last comma and newline
            ddl = ddl.rstrip(",\n") + "\n);\n"
            ddl_statements.append(ddl)
            
        return "\n".join(ddl_statements) 