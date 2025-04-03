import sqlite3
import pandas as pd
import chardet
from typing import List, Dict, Tuple, Any
from pathlib import Path
from .schema import ColumnSchema, TableSchema, DatabaseSchema

class SchemaExtractor:
    """Extractor of schema information from SQLite database"""
    
    @staticmethod
    def _load_database_description(db_folder: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load database description information
        :param db_folder: path to the database folder
        :return: database description information
        """
        description_dir = db_folder / "database_description"
        if not description_dir.exists():
            return {}
            
        database_description = {}
        for csv_file in description_dir.glob("*.csv"):
            table_name = csv_file.stem.lower()
            
            # Detect file encoding
            encoding = chardet.detect(csv_file.read_bytes())["encoding"]
            
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                table_description = {}
                
                for _, row in df.iterrows():
                    if pd.isna(row.get("original_column_name", "")):
                        continue
                        
                    col_name = row["original_column_name"].strip().lower()
                    table_description[col_name] = {
                        "expanded_name": row.get("column_name", "").strip() if pd.notna(row.get("column_name", "")) else "",
                        "description": row.get("column_description", "").strip() if pd.notna(row.get("column_description", "")) else "",
                        "value_description": row.get("value_description", "").strip() if pd.notna(row.get("value_description", "")) else "",
                        "data_format": row.get("data_format", "").strip() if pd.notna(row.get("data_format", "")) else ""
                    }
                    
                database_description[table_name] = table_description
            except Exception as e:
                print(f"Warning: Failed to load description file {csv_file}: {str(e)}")
                continue
                
        return database_description
    
    @staticmethod
    def _get_column_examples(conn: sqlite3.Connection, 
                          table: str, 
                          column: str, 
                          max_examples: int = 3) -> List[str]:
        """
        Get examples of column values
        :param conn: database connection
        :param table: table name
        :param column: column name
        :param max_examples: maximum number of examples
        :return: list of examples
        """
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT DISTINCT `{column}` FROM `{table}` "
                f"WHERE `{column}` IS NOT NULL AND `{column}` != '' "
                f"LIMIT {max_examples}"
            )
            return [str(row[0]) for row in cursor.fetchall()]
        except:
            return []
    
    @staticmethod
    def extract_from_db(db_path: str, db_id: str) -> DatabaseSchema:
        """Extract schema information from database file"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Load additional description information
        db_folder = Path(db_path).parent
        descriptions = SchemaExtractor._load_database_description(db_folder)
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]
        
        tables = {}
        all_foreign_keys = []
        
        # Process each table
        for table_name in table_names:
            # Get table structure
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
            
            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
            fk_info = cursor.fetchall()
            
            for fk in fk_info:
                id, seq, ref_table, from_col, to_col, *_ = fk
                
                # Update column foreign key information
                if from_col in columns:
                    columns[from_col].foreign_keys.append((ref_table, to_col))
                
                # Add to global foreign key list
                all_foreign_keys.append({
                    "table": [table_name, ref_table],
                    "column": [from_col, to_col]
                })
            
            # Add additional information
            table_desc = descriptions.get(table_name.lower(), {})
            for col_name, column in columns.items():
                col_desc = table_desc.get(col_name.lower(), {})
                column.expanded_name = col_desc.get("expanded_name", "")
                column.description = col_desc.get("description", "")
                column.value_description = col_desc.get("value_description", "")
                
                # Get examples of column values
                if column.type.upper() != "BLOB":
                    column.value_examples = SchemaExtractor._get_column_examples(conn, table_name, col_name)
            
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