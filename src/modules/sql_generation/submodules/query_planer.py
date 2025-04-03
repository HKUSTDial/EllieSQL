from typing import Dict, List
from ....core.llm import LLMBase
from ....core.sql_execute import *
from ....core.utils import load_json, load_jsonl
from ..prompts.query_plan_cot_prompts import SQL_GENERATION_SYSTEM, QUERY_PLAN_PROMPT, QUERY_PLAN_PROMPT2, EXAMPLE, QUERY_PLAN_GEN, SQL_GEN
from ..base import ModuleBase
from typing import Any, Dict, Optional, Tuple, Callable



class QueryPlaner():
    """QP method to generate SQL, used when ensembling"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 5000,
                module_name: str = "EnhancedSQLGenerator"):
        # super().__init__("FeedbackBasedRefiner")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.module_name = module_name


    # async def generate_sql_new(self, query: str, formatted_schema: str, curr_evidence: str)-> str:
    #     ##  1. Generate query plan
    #     messages = [
    #         {"role": "system", "content": SQL_GENERATION_SYSTEM},
    #         {"role": "user", "content": QUERY_PLAN_GEN.format(
    #             schema=formatted_schema,
    #             query=query,
    #             evidence=curr_evidence if curr_evidence else "None"
    #         )}
    #     ]

    #     result = await self.llm.call_llm(
    #         messages,
    #         self.model,
    #         temperature=self.temperature,
    #         max_tokens=self.max_tokens,
    #         module_name=self.module_name
    #     )

    #     query_plan = result["response"]
    #     print(query_plan)

    #     ##  2. Generate SQL based on query plan
    #     messages = [
    #         {"role": "system", "content": SQL_GENERATION_SYSTEM},
    #         {"role": "user", "content": SQL_GEN.format(
    #             schema=formatted_schema,
    #             query=query,
    #             evidence=curr_evidence if curr_evidence else "None",
    #             query_plan = query_plan
    #         )}
    #     ]

    #     result = await self.llm.call_llm(
    #         messages,
    #         self.model,
    #         temperature=self.temperature,
    #         max_tokens=self.max_tokens,
    #         module_name=self.module_name
    #     )

    #     return result






    async def generate_sql(self, query: str, formatted_schema: str, curr_evidence: str)-> str:
        """Generate SQL"""
        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": QUERY_PLAN_PROMPT.format(
                schema=formatted_schema,
                query=query,
                evidence=curr_evidence if curr_evidence else "None"
            )}
            # {"role": "user", "content": QUERY_PLAN_PROMPT2.format(
            #     example = EXAMPLE,
            #     schema=formatted_schema,
            #     query=query,
            #     evidence=curr_evidence if curr_evidence else "None"
            # )}
        ]
        
        # print('\n'+messages[1]['content']+'\n')
        result = await self.llm.call_llm(
            messages,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.module_name
        )

        #print(result["response"])
        
        # print("QP completed candidate SQL generation")
        return result

        # raw_output = result["response"]
        
        
        
        
