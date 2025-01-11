import re
from typing import Dict, List
# from ..base import PostProcessorBase
from ....core.llm import LLMBase
from ....core.sql_execute import *
from ....core.utils import load_json, load_jsonl
from ..prompts.refiner_prompts import REFINER_SYSTEM, REFINER_USER


from ..base import ModuleBase
from typing import Any, Dict, Optional, Tuple, Callable

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
                max_tokens: int = 5000):
        super().__init__("FeedbackBasedRefiner")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def process_sql(self, sql: str, query_id: str) -> str:
        """
        对生成的SQL进行运行、自反思检查和优化
        
        Args:
            sql: 原始SQL语句
            query_id: 查询ID，用于关联中间结果
        """
        # 加载SQL生成的结果
        
        # prev_result = self.load_previous_result(query_id)
        # original_query = prev_result["input"]["query"]

        merge_dev_demo_file = "./data/merge_dev_demo.json"
        merge_dev_demo_data = load_json(merge_dev_demo_file)

        for item in merge_dev_demo_data:
            if(item.get("question_id") == query_id):
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                question = item.get("question", "")
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
                    {"role": "user", "content": REFINER_USER.format(sql = sql, 
                                                                    result_type = ex_result.result_type, 
                                                                    result = ex_result.result, 
                                                                    error_message = ex_result.error_message , 
                                                                    question = question,
                                                                    db_schema = formatted_schema
                                                                    )}
                ]
                
                result = await self.llm.call_llm(
                    messages, 
                    self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    module_name=self.name
                )
                
                raw_output = result["response"]
                #processed_sql = self.extractor.extract_sql(raw_output)
                
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
                return raw_output
        