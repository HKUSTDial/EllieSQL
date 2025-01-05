from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class ColumnSchema:
    """
    Schema for a column in a database table.
    
    Attributes:
        original_column_name: The original column name.
        expanded_column_name: The expanded column name.
        column_type: The type of the column.
        column_description: The description of the column.
        value_description: The description of the values in the column.
        value_examples: Examples of the values in the column.
        primary_key: Whether the column is a primary key.
        foreign_keys: A list of foreign keys refecencing other tables.
        referenced_by: A list of columns in other tables that reference the column.
    """
    original_column_name: str = ""
    expanded_column_name: str = ""
    column_type: str = ""
    column_description: str = ""
    value_description: str = ""
    value_examples: List[str] = field(default_factory=list)
    primary_key: bool = False
    foreign_keys: List[Tuple[str, str]] = field(default_factory=list)
    referenced_by: List[Tuple[str, str]] = field(default_factory=list)
    
    @classmethod
    def from_column_schema_dict(cls, column_schema_dict: Dict[str, Any]) -> "ColumnSchema":
        """
        Create a ColumnSchema object from a dictionary.
        """
        return cls(
            original_column_name=column_schema_dict.get("original_column_name", ""),
            expanded_column_name=column_schema_dict.get("expanded_column_name", ""),
            column_type=column_schema_dict.get("column_type", ""),
            column_description=column_schema_dict.get("column_description", ""),
            value_description=column_schema_dict.get("value_description", ""),
            value_examples=column_schema_dict.get("value_examples", []),
            primary_key=column_schema_dict.get("primary_key", False),
            foreign_keys=column_schema_dict.get("foreign_keys", []),
            referenced_by=column_schema_dict.get("referenced_by", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ColumnSchema object to a dictionary.
        """
        return asdict(self)
    
@dataclass
class TableSchema:
    """
    Table schema for a database table.
    
    Attributes:
        table_name: The table name.
        columns: A dictionary mapping column names to their ColumnSchema objects.
    """
    table_name: str = ""
    columns: Dict[str, ColumnSchema] = field(default_factory=dict)

    def get_primary_keys(self) -> List[ColumnSchema]:
        """
        Get the primary key columns.
        """
        return [column for column in self.columns.values() if column.primary_key]
    
    @classmethod
    def from_table_schema_dict(cls, table_schema_dict: Dict[str, Dict[str, Any]]) -> "TableSchema":
        """
        Create a TableSchema object from a dictionary.
        """
        return cls(
            table_name=table_schema_dict.get("table_name", ""),
            columns={column_name: ColumnSchema.from_column_schema_dict(column_schema_dict) for column_name, column_schema_dict in table_schema_dict.get("columns", {}).items()},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the TableSchema object to a dictionary.
        """
        return asdict(self)

@dataclass
class DatabaseSchema:
    """
    Database schema for a database.
    
    Attributes:
        db_id: The database id.
        tables: A dictionary mapping table names to their TableSchema objects.
    """
    db_id: str = ""
    db_directory: str = ""
    tables: Dict[str, TableSchema] = field(default_factory=dict)
    
    @classmethod
    def from_database_schema_dict(cls, database_schema_dict: Dict[str, Dict[str, Dict[str, Any]]]) -> "DatabaseSchema":
        """
        Create a DatabaseSchema object from a dictionary.
        """
        return cls(
            db_id=database_schema_dict.get("db_id", ""),
            db_directory=database_schema_dict.get("db_directory", ""),
            tables={table_name: TableSchema.from_table_schema_dict(table_schema_dict) for table_name, table_schema_dict in database_schema_dict.get("tables", {}).items()},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DatabaseSchema object to a dictionary.
        """
        return asdict(self)
