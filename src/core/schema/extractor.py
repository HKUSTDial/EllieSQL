import sqlite3
from typing import List, Dict, Tuple
from pathlib import Path
from .schema import ColumnSchema, TableSchema, DatabaseSchema
from .utils import load_database_description, get_column_examples

class SchemaExtractor:
    """从SQLite数据库提取schema信息"""
    
    @staticmethod
    def extract_from_db(db_path: str, db_id: str) -> DatabaseSchema:
        """
        从数据库文件提取schema
        
        Args:
            db_path: 数据库文件路径
            db_id: 数据库ID
            
        Returns:
            DatabaseSchema: 提取的schema信息
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 加载补充描述信息
        db_folder = Path(db_path).parent
        descriptions = load_database_description(db_folder)
        
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]
        
        tables = {}
        all_foreign_keys = []
        
        for table_name in table_names:
            # 获取表结构
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns_info = cursor.fetchall()
            
            columns = {}
            primary_keys = []
            
            for col in columns_info:
                col_id, col_name, col_type, not_null, default_value, is_pk = col
                
                if is_pk:
                    primary_keys.append(col_name)
                    
                columns[col_name] = ColumnSchema(
                    name=col_name,
                    type=col_type,
                    is_primary_key=bool(is_pk)
                )
            
            # 获取外键信息
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
            fk_info = cursor.fetchall()
            
            for fk in fk_info:
                id, seq, ref_table, from_col, to_col, *_ = fk
                
                # 更新列的外键信息
                if from_col in columns:
                    columns[from_col].foreign_keys.append((ref_table, to_col))
                
                # 添加到全局外键列表
                all_foreign_keys.append({
                    "table": [table_name, ref_table],
                    "column": [from_col, to_col]
                })
            
            # 添加补充信息
            table_desc = descriptions.get(table_name.lower(), {})
            for col_name, column in columns.items():
                col_desc = table_desc.get(col_name.lower(), {})
                column.expanded_name = col_desc.get("expanded_name", "")
                column.description = col_desc.get("description", "")
                column.value_description = col_desc.get("value_description", "")
                
                # 获取示例值
                if column.type.upper() != "BLOB":
                    column.value_examples = get_column_examples(conn, table_name, col_name)
            
            tables[table_name] = TableSchema(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys
            )
        
        conn.close()
        
        return DatabaseSchema(
            database=db_id,
            tables=tables,
            foreign_keys=all_foreign_keys
        ) 