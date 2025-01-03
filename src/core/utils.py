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
        """从文本中提取JSON代码块并验证格式"""
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if not matches:
            try:
                schema = eval(text.strip())
            except:
                return None
        else:
            try:
                schema = eval(matches[0].strip())
            except:
                return None
                
        # 验证基本结构
        if not isinstance(schema, dict) or "tables" not in schema:
            return None
        if not isinstance(schema["tables"], list):
            return None
            
        # 验证每个表的格式
        for table in schema["tables"]:
            if not isinstance(table, dict):
                return None
            # 验证必需字段
            if not all(k in table for k in ["table", "columns"]):
                return None
            if not isinstance(table["table"], str):
                return None
            if not isinstance(table["columns"], list):
                return None
            if not all(isinstance(col, str) for col in table["columns"]):
                return None
                
            # 验证可选字段
            if "primary_keys" in table:
                if not isinstance(table["primary_keys"], list):
                    return None
                if not all(isinstance(pk, str) for pk in table["primary_keys"]):
                    return None
                    
            if "foreign_keys" in table:
                if not isinstance(table["foreign_keys"], list):
                    return None
                for fk in table["foreign_keys"]:
                    if not isinstance(fk, dict):
                        return None
                    if not all(k in fk for k in ["column", "referenced_table", "referenced_column"]):
                        return None
                    if not all(isinstance(fk[k], str) for k in ["column", "referenced_table", "referenced_column"]):
                        return None
                        
        return schema 