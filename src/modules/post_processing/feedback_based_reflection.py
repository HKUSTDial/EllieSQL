from typing import Dict, List
from .base import PostProcessorBase
from ...core.llm import LLMBase
from .prompts.reflection_prompts import REFLECTION_SYSTEM, FEEDBACK_BASED_REFLECTION_USER
import re

from ...core.sql_execute import *
from ...core.utils import load_json, load_jsonl

class FeedbackBasedReflectionPostProcessor(PostProcessorBase):
    """基于执行结果的使用自反思机制的SQL后处理器，使用SQL运行信息"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 1000):
        super().__init__("FeedbackBasedReflectionPostProcessor")
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
        
        prev_result = self.load_previous(query_id, "GPTSQLGenerator")
        original_query = prev_result["input"]["query"]

        merge_dev_demo_file = "./data/sampled_merged.json"
        merge_dev_demo_data = load_json(merge_dev_demo_file)

        for item in merge_dev_demo_data:
            if(item.get("question_id") == query_id):
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                question = item.get("question", "")
                db_path = "./data/merged_databases/" + source +'_'+ db_id +"/"+ db_id + '.sqlite'
                #执行sql并且返回结果：能运行、超时、或报错
                ex_result = execute_sql_with_timeout(db_path, sql)
                # if ex_result.result is None:
                #     if ex_result.result_type == SQLExecutionResultType.TIMEOUT:
                #         print(f"generated SQL 运行超时。错误信息如下：{ex_result.error_message}")
                #     elif ex_result.result_type == SQLExecutionResultType.ERROR:
                #         print(f"generated SQL 运行出现报错。错误信息如下：{ex_result.error_message}")
                #     elif ex_result.result_type == SQLExecutionResultType.ERROR:
                #         print(f"generated SQL 运行成功但结果为空。")
        
                messages = [
                    {"role": "system", "content": REFLECTION_SYSTEM},
                    {"role": "user", "content": FEEDBACK_BASED_REFLECTION_USER.format(sql=sql, ex_result=ex_result, question = question)}
                ]
                
                result = await self.llm.call_llm(
                    messages, 
                    self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    module_name=self.name
                )
                
                raw_output = result["response"]
                processed_sql = self.extractor.extract_sql(raw_output)
                
                # 保存中间结果
                self.save_intermediate(
                    input_data={
                        "original_query": original_query,
                        "original_sql": sql
                    },
                    output_data={
                        "raw_output": raw_output,
                        "processed_sql": processed_sql
                    },
                    model_info={
                        "model": self.model,
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "total_tokens": result["total_tokens"]
                    },
                    query_id=query_id
                )
                
                self.log_io({"original_sql": sql, "messages": messages}, processed_sql)
                return raw_output
        