from typing import Dict, List, Optional
from .base import SQLGeneratorBase
from ...core.llm import LLMBase
from .prompts.gpt_prompts import SQL_GENERATION_SYSTEM
from .prompts.direct_dc_prompt import DIRECT_DC_WITH_OS_PROMPT
from ...core.utils import load_json
from .submodules.refiner import *
from .submodules.online_synthesiser import *


class DirectDCOSRefineSQLGenerator(SQLGeneratorBase):
    """Implementation of using direct_dc prompt to generate SQL, with online synthesis examples, with a refiner (direct_dc + OS + refiner)"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 1000,
                max_retries: int = 3):
        super().__init__("DirectDCOSRefineSQLGenerator", max_retries)
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
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
        
        data_file = self.data_file
        dataset_examples = load_json(data_file)
        curr_evidence = ""
        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                curr_evidence = item.get("evidence", "")
                break


        # Record the token statistics for each step
        step_tokens = {
            "online_synthesis": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "direct_dc": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "refine": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }


        # 1. 先online synthesis
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

        # Build the prompt
        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": DIRECT_DC_WITH_OS_PROMPT.format(
                OS_EXAMPLES=examples_result["response"],
                SCHEMA_CONTEXT=formatted_schema,
                HINT=curr_evidence if curr_evidence else "None",
                QUESTION=query
            )}
        ]
        
        # print('\n'+messages[1]['content']+'\n')
        result = await self.llm.call_llm(
            messages,
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            module_name=self.name
        )

        step_tokens["direct_dc"]["input_tokens"] = result["input_tokens"]
        step_tokens["direct_dc"]["output_tokens"] = result["output_tokens"]
        step_tokens["direct_dc"]["total_tokens"] = result["total_tokens"]
        
        raw_output_before_refine = result["response"]

        extracted_sql = self.extractor.extract_sql_xml(raw_output_before_refine)
        
        # Refine stage
        refiner = FeedbackBasedRefiner(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)


        refine_result = await refiner.process_sql(extracted_sql, data_file, query_id)

        step_tokens["refine"]["input_tokens"] = refine_result["input_tokens"]
        step_tokens["refine"]["output_tokens"] = refine_result["output_tokens"]
        step_tokens["refine"]["total_tokens"] = refine_result["total_tokens"]

        refine_false = refine_result["refine_false"]
        refiner_raw_output = refine_result["response"] # This is the raw output after refining
        extracted_sql = self.extractor.extract_sql(refiner_raw_output)


        # Calculate the total token
        total_tokens = {
            "input_tokens": sum(step["input_tokens"] for step in step_tokens.values()),
            "output_tokens": sum(step["output_tokens"] for step in step_tokens.values()),
            "total_tokens": sum(step["total_tokens"] for step in step_tokens.values())
        }

        # Save the intermediate result
        self.save_intermediate(
            input_data={
                "query": query,
                "formatted_schema": formatted_schema,
                # "messages": messages
            },
            output_data={
                "raw_output": refiner_raw_output,
                "extracted_sql": extracted_sql,
                "refine_false": refine_false
            },
            model_info={
                "model": self.model,
                "input_tokens": total_tokens["input_tokens"],
                "output_tokens": total_tokens["output_tokens"],
                "total_tokens": total_tokens["total_tokens"]
            },
            query_id=query_id,
            module_name=self.name if (module_name == None) else module_name
        )
        
        self.log_io(
            input_data={
                "query": query, 
                "formatted_schema": formatted_schema,
                # "messages": messages
            }, 
            output_data=refiner_raw_output
        )
        
        return refiner_raw_output
        