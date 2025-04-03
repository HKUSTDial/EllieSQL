from typing import Dict, List
from ....core.llm import LLMBase
from ....core.sql_execute import *
from ....core.utils import load_json, load_jsonl
from ..prompts.refiner_prompts import REFINER_SYSTEM, REFINER_USER, REFINER_ALL_USER
from ....core.utils import TextExtractor
from ....core.config import Config

from src.core.schema.manager import SchemaManager


class RefinerBase():
    """后处理模块的基类"""
    def __init__(self, name: str):
        self.name = name
    
    # async def process_sql_with_retry(self, 
    #                               sql: str,
    #                               query_id: str = None) -> Tuple[str, str]:
    #     return await self.execute_with_retry(
    #         func=self.process_sql,
    #         extract_method='sql',
    #         error_msg="SQL Refiner失败",
    #         sql=sql,
    #         query_id=query_id
    #     ) 


class FeedbackBasedRefiner(RefinerBase):
    """SQL Refiner using self-reflection mechanism based on SQL execution results, using SQL execution information and DB Schema"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 5000,
                module_name: str = "EnhancedSQLGenerator"):
        super().__init__("FeedbackBasedRefiner")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.module_name = module_name
        

    async def process_sql(self, sql: str, data_file: str, query_id: str) -> str:
        """
        Run, self-reflect, and optimize the generated SQL
        (Only refine SQLs that execute with errors or result in NULL)
        """
        
        dataset_examples = load_json(data_file)
        db_path = None
        curr_sql = sql
        formatted_schema = None
        question = None
        curr_evidence = None
        
        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                question = item.get("question", "")
                curr_evidence = item.get("evidence")
                db_folder = f"{source}_{db_id}"
                db_file = f"{db_id}.sqlite"
                db_path = str(Config().database_dir / db_folder / db_file)

                # Get schema
                schema_manager = SchemaManager()
                formatted_schema = schema_manager.get_formatted_enriched_schema(db_id, source)
                break
        
        if db_path is None:
            raise ValueError(f"Could not find query_id {query_id} in dataset")

        extractor = TextExtractor()
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        flag = False
        iter_cnt = 0
        while(flag == False and iter_cnt < 3):
            ex_result = execute_sql_with_timeout(db_path, curr_sql)
            flag, error_message = validate_sql_execution(db_path, curr_sql)
            
            if(flag == True):
                result = {
                    "response": f"After {iter_cnt} refinements, the SQL executed successfully, directly returning the result ```sql {curr_sql}```",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "refine_false": 0
                }
                return result

            messages = [
                {"role": "system", "content": REFINER_SYSTEM},
                {"role": "user", "content": REFINER_USER.format(
                    sql = curr_sql, 
                    result_type = ex_result.result_type, 
                    result = ex_result.result, 
                    error_message = error_message,
                    question = question,
                    evidence = curr_evidence if curr_evidence else "None",
                    db_schema = formatted_schema
                )}
            ]
            
            result = await self.llm.call_llm(
                messages, 
                self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.module_name
            )

            input_tokens += result["input_tokens"]
            output_tokens += result["output_tokens"]
            total_tokens += result["total_tokens"]

            raw_output = result["response"]
            curr_sql = extractor.extract_sql(raw_output)
            iter_cnt += 1
        
        # Check the final result
        flag, error_message = validate_sql_execution(db_path, curr_sql)
        if flag:
            result = {
                "response": f"After {iter_cnt} refinements, the SQL executed successfully, directly returning the result ```sql {curr_sql}```",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "refine_false": 0
            }
        else:
            result = {
                "response": f"After {iter_cnt} refinements, the SQL executed with errors, directly returning the result ```sql {curr_sql}```",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "refine_false": 1
            }
        return result

    async def process_all_sql(self, sql: str, data_file: str, query_id: str) -> str:
        """Run, self-reflect, and optimize the generated SQL"""
        
        dataset_examples = load_json(data_file)

        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                question = item.get("question", "")
                curr_evidence = item.get("evidence")
                db_folder = f"{source}_{db_id}"
                db_file = f"{db_id}.sqlite"
                db_path = str(Config().database_dir / db_folder / db_file)
                # Execute SQL and return result: can run, timeout, or error
                ex_result = execute_sql_with_timeout(db_path, sql)

                # Get schema
                # In any place that requires getting schema
                from src.core.schema.manager import SchemaManager
                # Get schema manager instance
                schema_manager = SchemaManager()
                # Get formatted schema string
                formatted_schema = schema_manager.get_formatted_enriched_schema(db_id,source)
                # print(formatted_schema) 
        
                messages = [
                    {"role": "system", "content": REFINER_SYSTEM},
                    {"role": "user", "content": REFINER_ALL_USER.format(sql = sql, 
                                                                    result_type = ex_result.result_type, 
                                                                    result = ex_result.result, 
                                                                    error_message = ex_result.error_message , 
                                                                    question = question,
                                                                    evidence = curr_evidence if curr_evidence else "None",
                                                                    db_schema = formatted_schema
                                                                    )}
                ]
                
                result = await self.llm.call_llm(
                    messages, 
                    self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    module_name=self.module_name  # Use the module name of the generator, so the statistics will include the generator
                )

                return result
                # raw_output = result["response"]
                # return raw_output

                # processed_sql = self.extractor.extract_sql(raw_output)
                
                # # Save intermediate results
                # self.save_intermediate(
                #     input_data={
                #         "original_query": original_query,
                #         "original_sql": sql
                #     },
                #     output_data={
                #         "raw_output": raw_output,
                #         "processed_sql": processed_sql
                #     },
                #     model_info={
                #         "model": self.model,
                #         "input_tokens": result["input_tokens"],
                #         "output_tokens": result["output_tokens"],
                #         "total_tokens": result["total_tokens"]
                #     },
                #     query_id=query_id
                # )
                
                # self.log_io({"original_sql": sql, "messages": messages}, processed_sql)
                
        