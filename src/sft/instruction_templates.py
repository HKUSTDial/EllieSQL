from typing import Dict

class PipelineClassificationTemplates:
    """Pipeline classification task prompt templates"""
    

    @staticmethod
    def get_classifier_template() -> str:
        """Get the prompt template for the classification head model"""
        return """
Database Schema Related to the Question:
{schema_str}
Schema Analysis:
{schema_analysis}
Question: {question}"""

#     @staticmethod
#     def get_classifier_template() -> str:
#         """Get the prompt template for the classification head model"""
#         return """
# Database Schema Related to the Question:
# {schema_str}
# Question: {question}"""

    @staticmethod
    def get_generator_template() -> str:
        """Get the prompt template for the generation model"""
        return """Task Description:
This is a pipeline selection task for text-to-SQL generation. Given a natural language question and its corresponding database schema, we need to classify which SQL generation pipeline is most suitable:
1. Basic: For simple queries involving basic operations
2. Intermediate: For medium complexity queries that benefit from divide-and-conquer strategy
3. Advanced: For complex queries requiring both divide-and-conquer and online synthesis

Schema Analysis:
{schema_analysis}

Question: {question}

Database Schema:
{schema_str}

Classification:"""

    @staticmethod
    def analyze_schema(schema: Dict) -> str:
        """Analyze the schema and generate a description"""
        tables = schema["tables"]
        total_tables = len(tables)
        total_columns = sum(len(table["columns"]) for table in tables)
        
        # table_details = []
        # for table in tables:
        #     cols = len(table["columns"])
            # pks = len(table.get("primary_keys", []))
            # table_details.append(
            #     f"Table '{table['table']}' has {cols} columns"
            #     f"{f' with {pks} primary key(s)' if pks else ''}"
            # )
        
        analysis = f"The database schema related to the user question contains {total_tables} table{'s' if total_tables > 1 else ''} "
        analysis += f"with a total of {total_columns} columns.\n"
        # analysis += "Schema details:\n- " + "\n- ".join(table_details)
            
        return analysis
        
    @staticmethod
    def format_schema(schema: Dict) -> str:
        """Format the schema as a string"""
        result = []
        for table in schema["tables"]:
            result.append(f"Table: {table['table']}; Columns: {', '.join(table['columns'])}\n")
            # result.append(f"Table: {table['table']}")
            # result.append(f"Columns: {', '.join(table['columns'])}")
            # if "primary_keys" in table:
            #     result.append(f"Primary keys: {', '.join(table['primary_keys'])}")
            result.append("")
            
        return "".join(result)
        
    @classmethod
    def create_classifier_prompt(cls, question: str, schema: Dict) -> str:
        """Create the input text for the classification head model"""
        return cls.get_classifier_template().format(
            schema_analysis=cls.analyze_schema(schema),
            question=question,
            schema_str=cls.format_schema(schema)
        )
        
    @classmethod
    def create_generator_prompt(cls, question: str, schema: Dict) -> str:
        """Create the input text for the generation model"""
        return cls.get_generator_template().format(
            schema_analysis=cls.analyze_schema(schema),
            question=question,
            schema_str=cls.format_schema(schema)
        ) 