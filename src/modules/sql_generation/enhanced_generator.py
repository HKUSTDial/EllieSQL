from typing import Dict, List
from .base import SQLGeneratorBase
from ...core.llm import LLMBase
from .prompts.divide_and_conquer_cot_prompts import SQL_GENERATION_SYSTEM, DIVIDE_PROMPT, CONQUER_PROMPT, ASSEMBLE_PROMPT
from .prompts.online_synthesis_cot_prompts import SQL_GENERATION_SYSTEM, ONLINE_SYNTHESIS_PROMPT
import re
from .submodules.refiner import *
from .submodules.online_synthesiser import *

class EnhancedSQLGenerator(SQLGeneratorBase):
    """使用Divide And Conquer CoT + Online synthesis生成SQL的实现"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 5000,
                max_retries: int = 3):
        super().__init__("EnhancedSQLGenerator", max_retries)
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # async def generate_online_synthesis_examples(self, formatted_schema: str) -> Dict:
    #     """
    #     Generate online synthesis examples
        
    #     Args:
    #         formatted_schema: The formatted schema string
            
    #     Returns:
    #         Dict: The result dictionary containing examples and token statistics
    #     """
    #     # Build the prompt
    #     online_synthesis_messages = [
    #         {"role": "system", "content": SQL_GENERATION_SYSTEM},
    #         {"role": "user", "content": ONLINE_SYNTHESIS_PROMPT.format(
    #             TARGET_DATABASE_SCHEMA=formatted_schema,
    #             k=2
    #         )}
    #     ]
        
    #     # Call LLM to get examples
    #     result = await self.llm.call_llm(
    #         online_synthesis_messages,
    #         self.model,
    #         temperature=self.temperature,
    #         max_tokens=self.max_tokens,
    #         module_name=self.name
    #     )
    #     print("Generate examples completed")
    #     return result  # Return the complete result dictionary
        
    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str, module_name: Optional[str] = None) -> str:
        """Generate SQL"""
        # Load the schema linking result
        if not schema_linking_output:
            prev_result = self.load_previous_result(query_id)
            formatted_schema = prev_result["output"]["formatted_linked_schema"]
        else:
            formatted_schema = schema_linking_output.get("formatted_linked_schema")
            if not formatted_schema:  # If the result directly passed in does not have the formatted schema
                prev_result = self.load_previous_result(query_id)
                formatted_schema = prev_result["output"]["formatted_linked_schema"]
        
        # print("schema linking completed, start divide")
        data_file = self.data_file
        dataset_examples = load_json(data_file)

        curr_evidence = ""
        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                curr_evidence = item.get("evidence", "")
                break
        # 1. divide: 
        # Decompose the original question Qu into a set of sub-questions Sq
        divide_prompt = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": DIVIDE_PROMPT.format(
                schema=formatted_schema,
                query=query,
                evidence = curr_evidence if curr_evidence else "None"
            )}
        ]
        
        # Record the token statistics for each step
        step_tokens = {
            "divide": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "online_synthesis": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "conquer": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "assemble": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "refine": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
        
        # 1. Divide stage
        divide_result = await self.llm.call_llm(
            divide_prompt,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        step_tokens["divide"]["input_tokens"] = divide_result["input_tokens"]
        step_tokens["divide"]["output_tokens"] = divide_result["output_tokens"]
        step_tokens["divide"]["total_tokens"] = divide_result["total_tokens"]
        
        raw_output = divide_result["response"]
        sub_questions = self.extractor.extract_sub_questions(raw_output)
        
        # Initialize an empty set Ssql to store partial SQL queries for each sub-question
        ssql = [] 
        #print(sub_questions)
        # print("divide end")

        # 2. conquer:
        ## online synthesis examples for few-shot
        # online_synthesiser=OnlineSynthesis(llm=self.llm, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)

        # Use the wrapped method to generate online synthesis examples
        online_synthesiser = OnlineSynthesiser(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name )
        examples_result = await online_synthesiser.generate_online_synthesis_examples(formatted_schema)
        step_tokens["online_synthesis"]["input_tokens"] = examples_result["input_tokens"]
        step_tokens["online_synthesis"]["output_tokens"] = examples_result["output_tokens"]
        step_tokens["online_synthesis"]["total_tokens"] = examples_result["total_tokens"]
        
        # for each sub-question qi in Sq
        for sub_question in sub_questions:
        # Generate a partial SQL query for each sub-question qi
            conquer_prompt = [
                {"role": "system", "content": SQL_GENERATION_SYSTEM},
                {"role": "user", "content": CONQUER_PROMPT.format(
                    schema=formatted_schema,
                    examples=examples_result["response"],  # Use the response field
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
                module_name=self.name
            )
            step_tokens["conquer"]["input_tokens"] += conquer_result["input_tokens"]
            step_tokens["conquer"]["output_tokens"] += conquer_result["output_tokens"]
            step_tokens["conquer"]["total_tokens"] += conquer_result["total_tokens"]
            
            raw_output = conquer_result["response"]
            extracted_sql = self.extractor.extract_sql(raw_output)
            ssql.append(extracted_sql)

        # print("Conquer completed, start assemble")

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

        # 4. Assemble stage
        assemble_result = await self.llm.call_llm(
            assemble_prompt,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        step_tokens["assemble"]["input_tokens"] = assemble_result["input_tokens"]
        step_tokens["assemble"]["output_tokens"] = assemble_result["output_tokens"]
        step_tokens["assemble"]["total_tokens"] = assemble_result["total_tokens"]
        
        raw_output = assemble_result["response"]
        extracted_sql = self.extractor.extract_sql(raw_output)

        # print("Completed the initial SQL generation")

        refiner = FeedbackBasedRefiner(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)

        # 5. Refine stage
        refine_result = await refiner.process_sql(extracted_sql, data_file, query_id)

        refiner_raw_output = refine_result["response"]
        extracted_sql = self.extractor.extract_sql(refiner_raw_output)

        

        step_tokens["refine"]["input_tokens"] = refine_result["input_tokens"]
        step_tokens["refine"]["output_tokens"] = refine_result["output_tokens"]
        step_tokens["refine"]["total_tokens"] = refine_result["total_tokens"]
        # print("sql refine completed")
      
        # Calculate the total token
        total_tokens = {
            "input_tokens": sum(step["input_tokens"] for step in step_tokens.values()),
            "output_tokens": sum(step["output_tokens"] for step in step_tokens.values()),
            "total_tokens": sum(step["total_tokens"] for step in step_tokens.values())
        }

        # Save intermediate results, including the statistics for each step
        self.save_intermediate(
            input_data={
                "query": query,
                "formatted_schema": formatted_schema,
                "sub_questions": sub_questions,
                "sub_sqls": ssql
            },
            output_data={
                "raw_output": raw_output,
                "extracted_sql": extracted_sql,
                "refined_sql": extracted_sql
            },
            model_info={
                "model": self.model,
                "input_tokens": total_tokens["input_tokens"],
                "output_tokens": total_tokens["output_tokens"],
                "total_tokens": total_tokens["total_tokens"],
                "step_tokens": step_tokens  # Add the statistics for each step
            },
            query_id=query_id,
            module_name=self.name if (module_name == None) else module_name
        )
        
        self.log_io(
            input_data={
                "query": query, 
                "formatted_schema": formatted_schema,
                "messages": assemble_prompt
            }, 
            output_data=raw_output
        )
        
        return refiner_raw_output
        