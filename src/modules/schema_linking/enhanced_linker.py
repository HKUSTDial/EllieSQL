from typing import Dict
from .base import SchemaLinkerBase
from ...core.llm import LLMBase
from .prompts.schema_prompts import ENHANCED_SCHEMA_SYSTEM, BASE_SCHEMA_USER
from ...core.schema.manager import SchemaManager
from ...core.utils import load_json
import os

class EnhancedSchemaLinker(SchemaLinkerBase):
    """使用增强schema信息的Schema Linker"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 1000,
                max_retries: int = 3):
        super().__init__("EnhancedSchemaLinker", max_retries)
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.schema_manager = SchemaManager()
        
    def _validate_linked_schema(self, linked_schema: Dict, database_schema: Dict) -> bool:
        """
        验证linked schema中的表和列是否都在原始schema中存在
        
        Args:
            linked_schema: Schema Linking生成的schema
            database_schema: 原始数据库schema
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 创建原始schema的表和列的集合，用于快速查找
            db_tables = {table["table"]: set(table["columns"].keys()) 
                        for table in database_schema["tables"]}
            
            # 验证每个linked table
            for table in linked_schema["tables"]:
                table_name = table["table"]
                
                # 验证表是否存在
                if table_name not in db_tables:
                    self.logger.warning(f"表 '{table_name}' 在原始schema中不存在")
                    return False
                
                # 验证列是否存在于对应的表中
                for col in table["columns"]:
                    if col not in db_tables[table_name]:
                        self.logger.warning(
                            f"列 '{col}' 在表 '{table_name}' 中不存在"
                        )
                        return False
                        
                # 验证主键（如果有）
                if "primary_keys" in table:
                    for pk in table["primary_keys"]:
                        if pk not in db_tables[table_name]:
                            self.logger.warning(
                                f"主键 '{pk}' 在表 '{table_name}' 中不存在"
                            )
                            return False
                            
                # 验证外键（如果有）
                if "foreign_keys" in table:
                    for fk in table["foreign_keys"]:
                        # 验证外键列是否存在
                        if fk["column"] not in db_tables[table_name]:
                            self.logger.warning(
                                f"外键列 '{fk['column']}' 在表 '{table_name}' 中不存在"
                            )
                            return False
                        
                        # 验证引用的表和列是否存在
                        ref_table = fk["referenced_table"]
                        ref_column = fk["referenced_column"]
                        
                        if ref_table not in db_tables:
                            self.logger.warning(
                                f"引用的表 '{ref_table}' 在原始schema中不存在"
                            )
                            return False
                            
                        if ref_column not in db_tables[ref_table]:
                            self.logger.warning(
                                f"引用的列 '{ref_column}' 在表 '{ref_table}' 中不存在"
                            )
                            return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"验证linked schema时发生错误: {str(e)}")
            return False
            
    def _enhance_linked_schema(self, linked_schema: Dict, database_schema: Dict) -> Dict:
        """
        增强linked schema，添加必要的主键、外键信息和补充信息
        1. 为每个相关的表添加必要的主键列
        2. 处理表之间的外键关系
        3. 为所有列添加补充信息

        Args:
            linked_schema: Schema Linking生成的schema
            database_schema: 原始数据库schema
            
        Returns:
            Dict: 增强后的schema
        """
        # 创建数据库表信息的映射，方便查找
        db_tables = {table["table"]: table for table in database_schema["tables"]}
        
        # 获取所有涉及的表名
        linked_table_names = [table["table"] for table in linked_schema["tables"]]
        
        # 第一步：为每个表添加必要的列（主键和外键）
        for table in linked_schema["tables"]:
            table_name = table["table"]
            db_table = db_tables[table_name]
            
            # 1.1 添加主键列
            if db_table["primary_keys"]:
                if "primary_keys" not in table:
                    table["primary_keys"] = db_table["primary_keys"]
                
                # 确保主键列在columns中
                for pk in db_table["primary_keys"]:
                    if pk not in table["columns"]:
                        table["columns"].append(pk)
                        self.logger.info(f"为表 '{table_name}' 添加主键列 '{pk}'")
            
        # 1.2 处理表之间的外键关系
        if len(linked_table_names) > 1:  # 只有当涉及多个表时才处理外键
            for fk in database_schema.get("foreign_keys", []):
                src_table = fk["table"][0]
                dst_table = fk["table"][1]
                src_col = fk["column"][0]
                dst_col = fk["column"][1]
                
                # 只处理涉及的表之间的外键关系
                if src_table in linked_table_names and dst_table in linked_table_names:
                    # 找到源表和目标表在linked_schema中的对象
                    src_table_obj = next(t for t in linked_schema["tables"] if t["table"] == src_table)
                    dst_table_obj = next(t for t in linked_schema["tables"] if t["table"] == dst_table)
                    
                    # 确保外键列在columns中
                    for table_obj, col_name in [(src_table_obj, src_col), (dst_table_obj, dst_col)]:
                        if col_name not in table_obj["columns"]:
                            table_obj["columns"].append(col_name)
                            table_name = table_obj["table"]
                            if table_obj == src_table_obj:
                                self.logger.info(f"为表 '{src_table}' 添加外键列 '{src_col}'")
                            else:
                                self.logger.info(f"为表 '{dst_table}' 添加被引用列 '{dst_col}'")
                    
                    # 添加外键关系信息
                    if "foreign_keys" not in src_table_obj:
                        src_table_obj["foreign_keys"] = []
                        
                    # 检查是否已存在相同的外键关系
                    fk_exists = any(
                        fk["column"] == src_col and 
                        fk["referenced_table"] == dst_table and 
                        fk["referenced_column"] == dst_col 
                        for fk in src_table_obj.get("foreign_keys", [])
                    )
                    
                    if not fk_exists:
                        src_table_obj["foreign_keys"].append({
                            "column": src_col,
                            "referenced_table": dst_table,
                            "referenced_column": dst_col
                        })
                        self.logger.info(
                            f"添加外键关系: {src_table}.{src_col} -> {dst_table}.{dst_col}"
                        )
        
        # 第二步：为所有列添加补充信息
        for table in linked_schema["tables"]:
            table_name = table["table"]
            db_table = db_tables[table_name]
            
            # 为所有列添加补充信息
            columns_info = {}
            for col in table["columns"]:  # 现在包含了所有列（原有的、主键和外键）
                if col in db_table["columns"]:
                    col_info = db_table["columns"][col]
                    columns_info[col] = {
                        "type": col_info["type"],
                        "expanded_name": col_info.get("expanded_name", ""),
                        "description": col_info.get("description", ""),
                        "data_format": col_info.get("data_format", ""),
                        "value_description": col_info.get("value_description", ""),
                        "value_examples": col_info.get("value_examples", [])
                    }
            table["columns_info"] = columns_info
        
        return linked_schema
    
    async def link_schema(self, query: str, database_schema: Dict, query_id: str = None) -> str:
        """
        执行schema linking
        
        Args:
            query: 用户的自然语言查询
            database_schema: 数据库schema信息
            query_id: 查询的唯一标识符
            
        Returns:
            str: 链接后的schema信息
        """
        # 使用schema manager的格式化方法
        schema_str = self.schema_manager.format_enriched_db_schema(database_schema)

        # 获取evidence
        data_file = self.data_file
        dataset_examples = load_json(data_file)
        curr_evidence = ""
        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                curr_evidence = item.get("evidence", "")
                break
        
        # 记录格式化后的schema以便检查
        self.logger.debug("格式化后的Schema信息:")
        self.logger.debug("\n" + schema_str)
        
        messages = [
            {"role": "system", "content": ENHANCED_SCHEMA_SYSTEM},
            {"role": "user", "content": BASE_SCHEMA_USER.format(
                schema_str=schema_str,
                query=query,
                evidence=curr_evidence if curr_evidence else "None"
            )}
        ]
        # print('\n'+messages[1]['content']+'\n')
        
        result = await self.llm.call_llm(
            messages,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        raw_output = result["response"]
        extracted_linked_schema = self.extractor.extract_schema_json(raw_output)
        
        # 验证extracted schema
        if not extracted_linked_schema or not self._validate_linked_schema(extracted_linked_schema, database_schema):
            raise ValueError("Schema linking结果验证失败：包含不存在的表或列")
        
        # print("****before enhance*****\n", extracted_linked_schema, "\n*********")
        
        # 增强schema信息: 添加必要的主键、外键信息和补充信息
        enhanced_linked_schema = self._enhance_linked_schema(extracted_linked_schema, database_schema)
        # print("****after enhance*****\n", enhanced_linked_schema, "\n*********")
        
        # 格式化linked schema
        formatted_linked_schema = self.schema_manager.format_linked_schema(enhanced_linked_schema)
        # print("****after format*****\n", formatted_linked_schema, "\n*********")

        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query, 
                # "database_schema": database_schema,
                # "formatted_schema": schema_str,
                # "messages": messages
            },
            output_data={
                "raw_output": raw_output,
                "extracted_linked_schema": extracted_linked_schema,
                "enhanced_linked_schema": enhanced_linked_schema,  # 添加增强后的schema
                "formatted_linked_schema": formatted_linked_schema
            },
            model_info={
                "model": self.model,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"]
            },
            query_id=query_id
        )
        
        self.log_io(
            {
                "query": query, 
                "database_schema": database_schema, 
                "formatted_schema": schema_str,
                "messages": messages
            }, 
            raw_output
        )
        
        # Save the linked schema result
        # Get source from the database_schema or a default value
        source = database_schema.get("source", "unknown")
        self.save_linked_schema_result(
            query_id=query_id,
            source=source,
            linked_schema={
                "database": database_schema.get("database", ""),
                "tables": enhanced_linked_schema.get("tables", [])
            }
        )
        
        return raw_output 

    # def save_linked_schema_result(self, query_id: str, source: str, linked_schema: Dict) -> None:
    #     """
    #     Save the linked schema result to a separate JSONL file.
        
    #     Args:
    #         query_id: Query ID
    #         source: Data source (e.g., 'bird_dev')
    #         linked_schema: The linked schema with primary/foreign keys
    #     """
    #     # Get the pipeline directory from the intermediate result handler
    #     pipeline_dir = self.intermediate.pipeline_dir
        
    #     # Define the output file path
    #     output_file = os.path.join(pipeline_dir, "linked_schema_results.jsonl")
        
    #     # Format the result in the specified structure
    #     result = {
    #         "query_id": query_id,
    #         "source": source,
    #         "database": linked_schema.get("database", ""),
    #         "tables": [
    #             {
    #                 "table": table["table"],
    #                 "columns": table["columns"]
    #             }
    #             for table in linked_schema.get("tables", [])
    #         ]
    #     }
        
    #     # Save to JSONL file
    #     with open(output_file, "a", encoding="utf-8") as f:
    #         f.write(json.dumps(result, ensure_ascii=False) + "\n")  