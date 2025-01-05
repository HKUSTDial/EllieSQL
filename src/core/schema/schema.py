from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class ColumnSchema:
    """列的schema定义"""
    name: str
    type: str
    description: str = ""
    expanded_name: str = ""  # 扩展的列名（更易理解的名称）
    value_description: str = ""  # 值的描述
    data_format: str = ""  # 数据格式
    value_examples: List[str] = field(default_factory=list)
    is_primary_key: bool = False
    foreign_keys: List[Tuple[str, str]] = field(default_factory=list)  # [(target_table, target_column)]
    referenced_by: List[Tuple[str, str]] = field(default_factory=list)  # 被哪些表引用
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
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
    """表的schema定义"""
    name: str
    columns: Dict[str, ColumnSchema]
    primary_keys: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "table": self.name,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
            "primary_keys": self.primary_keys
        }

@dataclass
class DatabaseSchema:
    """数据库schema定义"""
    database: str
    tables: Dict[str, TableSchema]
    foreign_keys: List[Dict[str, List[str]]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
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
        生成数据库的DDL语句
        
        Args:
            add_expanded_name: 是否添加扩展列名
            add_description: 是否添加列描述
            add_value_description: 是否添加值描述
            add_examples: 是否添加示例值
            
        Returns:
            str: DDL语句
        """
        ddl_statements = []
        for table in self.tables.values():
            ddl = f"CREATE TABLE `{table.name}` (\n"
            
            # 添加列定义
            for col_name, column in table.columns.items():
                col_def = f"\t`{col_name}` {column.type}"
                
                # 添加注释
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
            
            # 添加主键约束
            if table.primary_keys:
                ddl += f"\tPRIMARY KEY ({', '.join(f'`{pk}`' for pk in table.primary_keys)}),\n"
            
            # 添加外键约束
            for col_name, column in table.columns.items():
                for fk_table, fk_column in column.foreign_keys:
                    ddl += f"\tFOREIGN KEY (`{col_name}`) REFERENCES `{fk_table}`(`{fk_column}`),\n"
            
            # 移除最后的逗号和换行符
            ddl = ddl.rstrip(",\n") + "\n);\n"
            ddl_statements.append(ddl)
            
        return "\n".join(ddl_statements) 