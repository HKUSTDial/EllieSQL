from typing import Dict, List
from ....core.llm import LLMBase
from ....core.sql_execute import *
from ....core.utils import load_json, load_jsonl
from ..prompts.online_synthesis_cot_prompts import SQL_GENERATION_SYSTEM, ONLINE_SYNTHESIS_PROMPT
from ..prompts.divide_and_conquer_cot_prompts import SQL_GENERATION_SYSTEM, DIVIDE_PROMPT, CONQUER_PROMPT_WO_EXAMPLES, ASSEMBLE_PROMPT
from ....core.utils import TextExtractor
from ..base import ModuleBase
from typing import Any, Dict, Optional, Tuple, Callable



class DivideConqueror():
    """DC方法生成SQL，ensemble时使用"""
    
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
        self.extractor = TextExtractor()


    async def generate_sql(self, query: str, formatted_schema: str, curr_evidence: str)-> str:
        """生成SQL"""
        # 1. Divide阶段
        # Decompose the original question Qu into a set of sub-questions Sq
        divide_prompt = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": DIVIDE_PROMPT.format(
                schema=formatted_schema,
                query=query,
                evidence = curr_evidence if curr_evidence else "None"
            )}
        ]
        
        divide_result = await self.llm.call_llm(
            divide_prompt,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.module_name
        )
        
        raw_output = divide_result["response"]
        sub_questions = self.extractor.extract_sub_questions(raw_output)
        
        # Initialize an empty set Ssql to store partial SQL queries for each sub-question
        ssql = [] 
        print("divide结束")
        # 2. conquer:
        # for each sub-question qi in Sq
        for sub_question in sub_questions:
        # Generate a partial SQL query for each sub-question qi
            conquer_prompt = [
                {"role": "system", "content": SQL_GENERATION_SYSTEM},
                {"role": "user", "content": CONQUER_PROMPT_WO_EXAMPLES.format(
                    schema=formatted_schema,
                    query=sub_question,
                    evidence = curr_evidence if curr_evidence else "None"
                )}
            ]
            # 3. Conquer阶段
            conquer_result = await self.llm.call_llm(
                conquer_prompt,
                self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.module_name
            )

            raw_output = conquer_result["response"]
            extracted_sql = self.extractor.extract_sql(raw_output)
            ssql.append(extracted_sql)

        print("Conquer 完成，开始assemble")
        # 3. assemble:
        # Assemble the final SQL query Sf from all sub-queries in Ssql
        sub_prompt = ""
        for i in range(len(sub_questions)):
            # sub_prompt += "Sub-question " + str(i) +": "
            sub_prompt += sub_questions[i] +"\n"
            sub_prompt += "SQL query " + str(i) +": "
            sub_prompt += ssql[i] +"\n\n"

        assemble_prompt = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": ASSEMBLE_PROMPT.format(
                schema=formatted_schema,
                query=query,
                evidence = curr_evidence if curr_evidence else "None",
                subs = sub_prompt
            )}
        ]
        # 3. Assemble阶段
        assemble_result = await self.llm.call_llm(
            assemble_prompt,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.module_name
        )

        # raw_output = assemble_result["response"]
        # extracted_sql = self.extractor.extract_sql(raw_output)

        print("DC完成了候选sql生成")

        return assemble_result
        

                
        