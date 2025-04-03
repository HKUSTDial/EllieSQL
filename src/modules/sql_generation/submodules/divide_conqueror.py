from typing import Dict, List
from ....core.llm import LLMBase
from ....core.sql_execute import *
from ....core.utils import load_json, load_jsonl
from ..prompts.divide_and_conquer_cot_prompts import SQL_GENERATION_SYSTEM, DIVIDE_PROMPT, CONQUER_PROMPT_WO_EXAMPLES, ASSEMBLE_PROMPT
from ....core.utils import TextExtractor
from ..base import ModuleBase
from typing import Any, Dict, Optional, Tuple, Callable



class DivideConqueror():
    """DC method to generate SQL, used when ensembling"""
    
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

        input_tokens = divide_result["input_tokens"]
        output_tokens = divide_result["output_tokens"]
        total_tokens = divide_result["total_tokens"]

        
        raw_output = divide_result["response"]
        sub_questions = self.extractor.extract_sub_questions(raw_output)

        # print(sub_questions)
        
        # Initialize an empty set Ssql to store partial SQL queries for each sub-question
        ssql = [] 
        
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
            # 3. Conquer stage
            conquer_result = await self.llm.call_llm(
                conquer_prompt,
                self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.module_name
            )


            input_tokens += conquer_result["input_tokens"]
            output_tokens += conquer_result["output_tokens"]
            total_tokens += conquer_result["total_tokens"]

            raw_output = conquer_result["response"]
            # extracted_sql = self.extractor.extract_sql(raw_output)
            # ssql.append(extracted_sql)
            ssql.append(raw_output)

        # print("Conquer completed, start assemble")
        
        # 3. assemble:
        # Assemble the final SQL query Sf from all sub-queries in Ssql
        sub_prompt = ""
        for i in range(len(sub_questions)):
            sub_prompt += "Sub-question " + str(i) +": "
            sub_prompt += sub_questions[i] +"\n"
            sub_prompt += "SQL query " + str(i) +": "
            sub_prompt += ssql[i] +"\n\n"

        # print(sub_prompt)
    
        assemble_prompt = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": ASSEMBLE_PROMPT.format(
                schema=formatted_schema,
                query=query,
                evidence = curr_evidence if curr_evidence else "None",
                subs = sub_prompt
            )}
        ]
        
        # 3. Assemble stage
        assemble_result = await self.llm.call_llm(
            assemble_prompt,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.module_name
        )

        # raw_output = assemble_result["response"]
        # extracted_sql = self.extractor.extract_sql(raw_output)

        assemble_result["input_tokens"] += input_tokens
        assemble_result["output_tokens"] += output_tokens
        assemble_result["total_tokens"] += total_tokens

        return assemble_result
        

                
        