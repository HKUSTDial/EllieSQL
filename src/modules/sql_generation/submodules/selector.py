from typing import Dict, List
from ....core.llm import LLMBase
from ....core.sql_execute import *
from ....core.utils import load_json, load_jsonl
from ..prompts.selector_prompts import SELECTOR_SYSTEM, SELECTOR_USER, CANDIDATE_FORMAT
from src.core.schema.manager import SchemaManager
from ....core.utils import TextExtractor
from ....core.config import Config

from ..base import ModuleBase
from typing import Any, Dict, Optional, Tuple, Callable


class DirectSelector():
    """基于prompt直接从多个候选中选择正确SQL的SQL Selector，使用SQL运行信息和DB Schema"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 5000,
                module_name: str = "EnhancedSQLGenerator"):
        # super().__init__("DirectSelector")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extractor = TextExtractor()
        self.module_name = module_name
        
    async def select_sql(self, data_file: str, sql_list: List[str], query_id: str) -> str:
        """
        对生成的多个SQL进行选择，选择一个
        
        Args:
            sql_list: 候选SQL列表
            query_id: 查询ID，用于关联中间结果
        """
       
        dataset_examples = load_json(data_file)
        db_path = None
        formatted_schema = None
        question = None
        evidence = None

        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                db_id = item.get("db_id", "")
                source = item.get("source", "")
                evidence = item.get("evidence")
                question = item.get("question", "")
                db_folder = f"{source}_{db_id}"
                db_file = f"{db_id}.sqlite"
                db_path = str(Config().database_dir / db_folder / db_file)

                # 获取schema
                schema_manager = SchemaManager()
                formatted_schema = schema_manager.get_formatted_enriched_schema(db_id, source)
                break
        
        if db_path is None:
            raise ValueError(f"Could not find query_id {query_id} in dataset")

        # 执行(多个)sql并且返回结果
        candidate_num = len(sql_list)
        ex_results = []
        for sql in sql_list:
            ex_result = execute_sql_with_timeout(db_path, sql)
            ex_results.append(ex_result)

        candidate_list = ""
        for i in range(candidate_num):
            candidate_list += CANDIDATE_FORMAT.format(
                i=i,
                isql=sql_list[i],
                iresult_type=str(ex_results[i].result_type),
                iresult=ex_results[i].result, 
                ierror_message=ex_results[i].error_message
            )
        
        user_content = SELECTOR_USER.format(
            candidate_num=candidate_num,
            db_schema=formatted_schema,
            evidence=evidence if evidence else "None",
            question=question,
            candidate_list=candidate_list
        )
         
        messages = [
            {"role": "system", "content": SELECTOR_SYSTEM},
            {"role": "user", "content": user_content}
        ]
        
        result = await self.llm.call_llm(
            messages, 
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.module_name
        )
        
        return result
        # 保存中间结果
        # self.save_intermediate(
        #     input_data={
        #         "question": question,
        #         "candidate_sqls": sql_list,
        #         "execution_results": [
        #             {
        #                 "sql": sql,
        #                 "result_type": str(ex_result.result_type),
        #                 "result": ex_result.result,
        #                 "error_message": ex_result.error_message
        #             }
        #             for sql, ex_result in zip(sql_list, ex_results)
        #         ]
        #     },
        #     output_data={
        #         "raw_output": raw_output,
        #         "selected_sql": selected_sql
        #     },
        #     model_info={
        #         "model": self.model,
        #         "input_tokens": result["input_tokens"],
        #         "output_tokens": result["output_tokens"],
        #         "total_tokens": result["total_tokens"]
        #     },
        #     query_id=query_id
        # )
        
        # self.log_io({"candidates": sql_list, "messages": messages}, selected_sql)
        
        