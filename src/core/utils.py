from typing import Dict, List, Optional
import re

import json

def load_json(file_path):
    """
    load json file.

    :param file_path: The path to the JSON file.
    :return: The data in the JSON file.
    """ 
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_jsonl(file_path):
    """
    load jsonl file.
    
    :param file_path: The path to the JSONL file.
    :return: The data in the JSONL file.
    """ 
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object
            data.append(json.loads(line.strip()))
    return data

class TextExtractor:
    """Text extraction tool class"""
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
        """Extract the SQL code block from the text, take the last ```sql``` block"""
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else text.strip()
    
    @staticmethod
    def extract_code_block(text: str, language: str = None) -> str:
        """
        Extract the content of the code block from the text, support specifying the language or extracting the code block of any language
        
        :param text: The text containing the code block
        :param language: The language identifier of the code block (optional), e.g. 'sql', 'json', 'python'
            
        :return: The extracted code block content (removed前后空白), if not found, return the original text
        """
        if language:
            # Extract the code block of the specified language
            pattern = f"```{language}\\s*(.*?)\\s*```"
        else:
            # Extract the code block of any language
            pattern = r"```\w*\s*(.*?)\s*```"
            
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else text.strip()
        
    @staticmethod
    def extract_schema_json(text: str) -> Optional[Dict]:
        """Extract the JSON code block from the text and verify the format"""
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
                
        # Verify the basic structure
        if not isinstance(schema, dict) or "tables" not in schema:
            return None
        if not isinstance(schema["tables"], list):
            return None
            
        # Verify the format of each table
        for table in schema["tables"]:
            if not isinstance(table, dict):
                return None
            # Verify the required fields
            if not all(k in table for k in ["table", "columns"]):
                return None
            if not isinstance(table["table"], str):
                return None
            if not isinstance(table["columns"], list):
                return None
            if not all(isinstance(col, str) for col in table["columns"]):
                return None
                
            # Verify the optional fields
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