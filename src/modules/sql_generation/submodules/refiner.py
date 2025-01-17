from typing import Dict, List
from ....core.llm import LLMBase
from ....core.sql_execute import *
from ....core.utils import load_json, load_jsonl
from ..prompts.refiner_prompts import REFINER_SYSTEM, REFINER_USER, REFINER_ALL_USER
from ....core.utils import TextExtractor

from src.core.schema.manager import SchemaManager


class RefinerBase():
    """后处理模块的基类"""
    def __init__(self, name: str):
        self.name = name
    
    # async def process_sql_with_retry(self, 
    #                               sql: str,
    #                               query_id: str = None) -> Tuple[str, str]:
    #     """使用重试机制处理SQL"""
    #     return await self.execute_with_retry(
    #         func=self.process_sql,
    #         extract_method='sql',
    #         error_msg="SQL Refiner失败",
    #         sql=sql,
    #         query_id=query_id
    #     ) 


class FeedbackBasedRefiner(RefinerBase):
    """基于执行结果的使用自反思机制的SQL Refiner，使用SQL运行信息和DB Schema"""
    
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
        """对生成的SQL进行运行、自反思检查和优化
        
        (只对执行出错和结果为空的sql进行refine)
        
        """
        
        dataset_examples = load_json(data_file)

        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                question = item.get("question", "")
                curr_evidence = item.get("evidence")
                db_path = "./data/merged_databases/" + source +'_'+ db_id +"/"+ db_id + '.sqlite'

                # 获取schema
                # 获取schema manager实例
                schema_manager = SchemaManager()
                # 获取格式化的schema字符串
                formatted_schema = schema_manager.get_formatted_enriched_schema(db_id,source)
                # print(formatted_schema) 
                extractor = TextExtractor()


                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

                curr_sql = sql


                flag = False
                iter_cnt = 0
                while(flag == False and iter_cnt < 3):
                    #执行sql并且返回结果：能运行、超时、或报错
                    ex_result = execute_sql_with_timeout(db_path, curr_sql)
                    flag, error_message= validate_sql(db_path, curr_sql)
                    # print(flag)
                    if(flag == True):
                        result = {
                            "response": f"sql经过{iter_cnt}次refine正常执行，直接返回结果 ```sql {curr_sql}```",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "refine_false": 0
                        }
                        return result

                    print("需要refine")
                    messages = [
                        {"role": "system", "content": REFINER_SYSTEM},
                        {"role": "user", "content": REFINER_USER.format(sql = curr_sql, 
                                                                        result_type = ex_result.result_type, 
                                                                        result = ex_result.result, 
                                                                        error_message = error_message , 
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
                        module_name=self.module_name  # 使用生成器的模块名，这样统计会计入生成器
                    )

                    input_tokens += result["input_tokens"]
                    output_tokens += result["output_tokens"]
                    total_tokens += result["total_tokens"]

                    raw_output = result["response"]
                    curr_sql = extractor.extract_sql(raw_output)
                    iter_cnt += 1
                
                flag, error_message = validate_sql(db_path, curr_sql)
                if(flag == True):
                    result = {
                                "response": f"sql迭代超过{iter_cnt}次后执行成功，直接返回 ```sql {curr_sql}```",
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": total_tokens,
                                "refine_false": 0
                            }
                    return result
                else:
                    result = {
                                "response": f"sql迭代超过{iter_cnt}次后执行出错，直接返回 ```sql {curr_sql}```",
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": total_tokens,
                                "refine_false": 1
                            }
                    # print(f"###qwert### sql迭代超过{iter_cnt}次后执行出错，直接返回")
                    # print(db_path)
                    # print(curr_sql)
                    return result

    async def process_all_sql(self, sql: str, data_file: str, query_id: str) -> str:
        """对生成的SQL进行运行、自反思检查和优化"""
        
        dataset_examples = load_json(data_file)

        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                question = item.get("question", "")
                curr_evidence = item.get("evidence")
                db_path = "./data/merged_databases/" + source +'_'+ db_id +"/"+ db_id + '.sqlite'
                #执行sql并且返回结果：能运行、超时、或报错
                ex_result = execute_sql_with_timeout(db_path, sql)

                # 获取schema
                # 在任何需要获取schema的地方
                from src.core.schema.manager import SchemaManager
                # 获取schema manager实例
                schema_manager = SchemaManager()
                # 获取格式化的schema字符串
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
                    module_name=self.module_name  # 使用生成器的模块名，这样统计会计入生成器
                )

                return result
                # raw_output = result["response"]
                # return raw_output

                # processed_sql = self.extractor.extract_sql(raw_output)
                
                # # 保存中间结果
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
                
        