from typing import Dict, List, Optional
import re

import json

def load_json(file_path):
    """
    load json file.

    Args: file_path
    Returns: file_data
    """ 
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_jsonl(file_path):
    """
    load jsonl file.
    
    Args: file_path
    Returns: file_data
    """ 
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的 JSON 对象
            data.append(json.loads(line.strip()))
    return data

class TextExtractor:
    """文本提取工具类"""
    @staticmethod
    def extract_sub_questions(text: str) -> List[str]:
        """
        Extracts sub-questions from the LLM response, assuming each sub-question is enclosed in <<>>.
        
        :param text: The response text from the model containing sub-questions
        :return: A list of sub-questions
        """
        # Regular expression to find text enclosed in <<>>
        pattern = r"<<(.*?)>>"
        sub_questions = re.findall(pattern, text, re.DOTALL)
        return sub_questions
    
    @staticmethod
    def extract_sql(text: str) -> str:
        """从文本中提取SQL代码块, 取最后一个```sql```块中的sql"""
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else text.strip()
    
    @staticmethod
    def extract_code_block(text: str, language: str = None) -> str:
        """
        从文本中提取代码块内容，支持指定语言或提取任意语言的代码块
        
        Args:
            text: 包含代码块的文本
            language: 代码块的语言标识(可选)，如'sql', 'json', 'python'等
            
        Returns:
            str: 提取的代码块内容(去除前后空白)，如果未找到则返回原文本
        """
        if language:
            # 提取指定语言的代码块
            pattern = f"```{language}\s*(.*?)\s*```"
        else:
            # 提取任意语言的代码块
            pattern = r"```\w*\s*(.*?)\s*```"
            
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else text.strip()
        
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