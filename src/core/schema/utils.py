import pandas as pd
import chardet
from pathlib import Path
from typing import Dict, Any, List, Optional
import sqlite3

def load_database_description(db_folder: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    加载数据库描述信息
    
    Args:
        db_folder: 数据库文件夹路径
        
    Returns:
        Dict: 数据库描述信息
    """
    description_dir = db_folder / "database_description"
    if not description_dir.exists():
        return {}
        
    database_description = {}
    for csv_file in description_dir.glob("*.csv"):
        table_name = csv_file.stem.lower()
        
        # 检测文件编码
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

def get_column_examples(conn: sqlite3.Connection, 
                       table: str, 
                       column: str, 
                       max_examples: int = 3) -> List[str]:
    """获取列的示例值"""
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