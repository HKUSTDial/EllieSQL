import re
from typing import Dict, List
from ..base import PostProcessorBase
from ....core.llm import LLMBase
from ....core.sql_execute import *
from ....core.utils import load_json, load_jsonl
from ..prompts.selector_prompts import SELECTOR_SYSTEM, SELECTOR_USER, CANDIDATE_FORMAT


from ..base import ModuleBase
from typing import Any, Dict, Optional, Tuple, Callable

class SelectorBase(ModuleBase):
    """Selector模块的基类"""
    pass
    # async def process_sql_with_retry(self, 
    #                               sql_list: List[str], #不知道这样修改是否合理
    #                               query_id: str = None) -> Tuple[str, str]:
    #     """使用重试机制处理SQL"""
    #     return await self.execute_with_retry(
    #         func=self.process_sql,
    #         extract_method='sql',
    #         error_msg="SQL Selector失败",
    #         sql_list=sql_list,
    #         query_id=query_id
    #     ) 


class DirectSelector(SelectorBase):
    """基于prompt直接从多个候选中选择正确SQL的SQL Selector，使用SQL运行信息和DB Schema"""
    
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
        
    async def select_sql(self, sql_list: List[str], query_id: str) -> str:
        """
        对生成的多个SQL进行选择，选择一个
        
        Args:
            sql: 原始SQL语句
            query_id: 查询ID，用于关联中间结果
        """
        # 加载SQL生成的结果
        
        prev_result = self.load_previous_result(query_id)
        original_query = prev_result["input"]["query"]

        merge_dev_demo_file = "./data/merge_dev_demo.json"
        merge_dev_demo_data = load_json(merge_dev_demo_file)

        for item in merge_dev_demo_data:
            if(item.get("question_id") == query_id):
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                evidence = item.get("evidence")
                question = item.get("question", "")
                db_path = "./data/merged_databases/" + source +'_'+ db_id +"/"+ db_id + '.sqlite'
                #执行(多个)sql并且返回结果：能运行、超时、或报错
                candidate_num = len(sql_list)
                ex_results = []
                for sql in sql_list:
                    ex_result = execute_sql_with_timeout(db_path, sql)
                    ex_results.append(ex_result)

                # 获取schema
                # 在任何需要获取schema的地方
                from src.core.schema.manager import SchemaManager
                # 获取schema manager实例
                schema_manager = SchemaManager()
                # 获取格式化的schema字符串
                formatted_schema = schema_manager.get_formatted_enriched_schema(db_id,source)

                user_content = SELECTOR_USER.format(
                                                    candidate_num = candidate_num,
                                                    db_schema = formatted_schema,
                                                    evidence = evidence,
                                                    question = question
                                                    )
                
                for i in range (candidate_num):
                    user_content += CANDIDATE_FORMAT.format(
                        i = i,
                        isql = sql_list[i],
                        iresult_type = ex_results[i].result_type, 
                        iresult = ex_results[i].result, 
                        ierror_message = ex_results[i].error_message
                        )
                     
                messages = [
                    {"role": "system", "content": SELECTOR_SYSTEM},
                    {"role": "user", "content":  user_content}
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
                return processed_sql
        