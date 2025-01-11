from typing import Dict, List
from .base import SQLGeneratorBase
from ...core.llm import LLMBase
from .prompts.divide_and_conquer_cot_prompts import SQL_GENERATION_SYSTEM, DIVIDE_PROMPT, CONQUER_PROMPT, ASSEMBLE_PROMPT
import re
from submodules.online_synthesis_cot_generator import OnlineSynthesis

class EnhancedSQLGenerator(SQLGeneratorBase):
    """使用Divide And Conquer CoT + Online synthesis生成SQL的实现"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 5000):
        super().__init__("EnhancedSQLGenerator")
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """生成SQL"""
        # 加载schema linking的结果
        if not schema_linking_output:
            prev_result = self.load_previous_result(query_id)
            formatted_schema = prev_result["output"]["formatted_linked_schema"]
        else:
            formatted_schema = schema_linking_output.get("formatted_linked_schema")
            if not formatted_schema:  # 如果直接传入的结果中没有格式化的schema
                prev_result = self.load_previous_result(query_id)
                formatted_schema = prev_result["output"]["formatted_linked_schema"]
        
        # 1. divide: 
        # Decompose the original question Qu into a set of sub-questions Sq
        divide_prompt = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": DIVIDE_PROMPT.format(
                schema=formatted_schema,
                query=query
            )}
        ]
        result = await self.llm.call_llm(
            divide_prompt,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        raw_output = result["response"]
        sub_questions = self.extractor.extract_sub_questions(raw_output)
        # Initialize an empty set Ssql to store partial SQL queries for each sub-question
        ssql = [] 
        # 2. conquer:
        ## online synthesis examples for few-shot
        online_synthesiser=OnlineSynthesis(llm=self.llm, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
        examples = online_synthesiser.generate_examples(query,
            formatted_schema,
            query_id=query_id,
            k=5
            )
        # for each sub-question qi in Sq
        for sub_question in sub_questions:
        # Generate a partial SQL query for each sub-question qi
            conquer_prompt = [
                {"role": "system", "content": SQL_GENERATION_SYSTEM},
                {"role": "user", "content": CONQUER_PROMPT.format(
                    schema=formatted_schema,
                    examples = examples,
                    query=sub_question
                )}
            ]
            result = await self.llm.call_llm(
                conquer_prompt,
                self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name
            )
            raw_output = result["response"]
            extracted_sql = self.extractor.extract_sql(raw_output)
            ssql.append(extracted_sql)
        # 3. assemble:
        # Assemble the final SQL query Sf from all sub-queries in Ssql
        sub_prompt = ""
        for i in len(sub_questions):
            sub_prompt += "Sub-question " + i +": "
            sub_prompt += sub_questions[i] +"\n"
            sub_prompt += "sql query " + i +": "
            sub_prompt += ssql[i]

        assemble_prompt = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": ASSEMBLE_PROMPT.format(
                schema=formatted_schema,
                query=query,
                subs = sub_prompt
            )}
        ]
        result = await self.llm.call_llm(
            assemble_prompt,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        raw_output = result["response"]
        extracted_sql = self.extractor.extract_sql(raw_output)
        
        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query,
                "formatted_schema": formatted_schema,
                "messages": assemble_prompt #这样修改合理吗
            },
            output_data={
                "raw_output": raw_output,
                "extracted_sql": extracted_sql
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
            input_data={
                "query": query, 
                "formatted_schema": formatted_schema,
                "messages": assemble_prompt #这样修改合理吗
            }, 
            output_data=raw_output
        )
        
        return raw_output
        