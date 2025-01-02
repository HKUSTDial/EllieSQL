from typing import Dict, Optional
import re

class TextExtractor:
    """文本提取工具类"""
    
    @staticmethod
    def extract_sql(text: str) -> str:
        """从文本中提取SQL代码块"""
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()
        
    @staticmethod
    def extract_schema_json(text: str) -> Optional[Dict]:
        """
        从文本中提取JSON代码块并验证格式
        
        预期的schema格式：
        {
            "tables": [
                {
                    "table": "table_name",
                    "columns": ["col1", "col2", ...]
                },
                ...
            ]
        }
        """
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if not matches:
            # 如果没有代码块，尝试直接解析整个文本
            try:
                schema = eval(text.strip())
            except:
                return None
        else:
            try:
                schema = eval(matches[0].strip())
            except:
                return None
                
        # 验证schema格式
        if not isinstance(schema, dict):
            return None
        if "tables" not in schema:
            return None
        if not isinstance(schema["tables"], list):
            return None
            
        # 验证每个表的格式
        for table in schema["tables"]:
            if not isinstance(table, dict):
                return None
            if "table" not in table or "columns" not in table:
                return None
            if not isinstance(table["table"], str):
                return None
            if not isinstance(table["columns"], list):
                return None
            if not all(isinstance(col, str) for col in table["columns"]):
                return None
            
        return schema 