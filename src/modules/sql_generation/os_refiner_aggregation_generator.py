from typing import Dict, List, Optional
from .online_synthesis_refiner_generator import OSRefinerSQLGenerator
from ...core.llm import LLMBase
from .prompts.aggregator_prompts import AGGREGATOR_SYSTEM, AGGREGATOR_USER
from ...core.utils import load_json
from .submodules.refiner import *

class OSRefinerAggregationSQLGenerator(OSRefinerSQLGenerator):
    """SQL generator that uses multiple OSRefiners to generate candidates and aggregate them, inheriting from OSRefinerSQLGenerator"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,  # The temperature used for aggregation
                max_tokens: int = 5000,
                n_candidates: int = 3,  # The number of candidates generated
                max_retries: int = 3):
        super().__init__(
            llm=llm,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        )
        self.name = "OSRefinerAggregationSQLGenerator"
        self.n_candidates = n_candidates
        
    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str, module_name: Optional[str] = None) -> str:
        """Generate multiple candidate SQLs and aggregate them"""
        # Record the token statistics for each step
        step_tokens = {
            "candidates": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "aggregator": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "refine": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
        
        # Load the schema linking result
        if not schema_linking_output:
            prev_result = self.load_previous_result(query_id)
            formatted_schema = prev_result["output"]["formatted_linked_schema"]
        else:
            formatted_schema = schema_linking_output.get("formatted_linked_schema")
            if not formatted_schema:
                prev_result = self.load_previous_result(query_id)
                formatted_schema = prev_result["output"]["formatted_linked_schema"]
                
        # Get evidence
        data_file = self.data_file
        dataset_examples = load_json(data_file)
        curr_evidence = ""
        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                curr_evidence = item.get("evidence", "")
                break
                
        # 1. Generate multiple candidate SQLs
        candidates = []
        candidates_gen_temperature = 1.0  # Use higher temperature to generate diverse candidates
        
        # Save the original temperature
        original_temperature = self.temperature
        
        try:
            # NOTE: Temporarily modify the temperature to a high value
            self.temperature = candidates_gen_temperature
            
            for i in range(self.n_candidates):
                # Call the generate_sql method of the parent class to generate candidates
                raw_output = await super().generate_sql(
                    query=query,
                    schema_linking_output=schema_linking_output,
                    query_id=query_id,
                    module_name=f"{self.name}_candidate_{i}"
                )
                # print('\nraw_output: ', raw_output)
                sql = self.extractor.extract_sql(raw_output)
                if sql:
                    candidates.append(sql)
                    
                # Add the token statistics for the candidates stage
                if hasattr(self, '_last_result'):
                    step_tokens["candidates"]["input_tokens"] += self._last_result["input_tokens"]
                    step_tokens["candidates"]["output_tokens"] += self._last_result["output_tokens"]
                    step_tokens["candidates"]["total_tokens"] += self._last_result["total_tokens"]
                    
        finally:
            # NOTE: Restore the original temperature
            self.temperature = original_temperature
                
        if not candidates:
            self.logger.error(f"Failed to generate valid candidate SQLs! Question ID: {query_id}")
            return "```sql\nSELECT 1;\n```"  # Return a simple fallback SQL
            
        # 2. Aggregate candidate SQLs (use the original low temperature)
        candidates_str = "\n".join(f"# Candidate {i+1}:\n{sql}" for i, sql in enumerate(candidates))
        
        messages = [
            {"role": "system", "content": AGGREGATOR_SYSTEM},
            {"role": "user", "content": AGGREGATOR_USER.format(
                schema=formatted_schema,
                query=query,
                evidence=curr_evidence,
                candidates=candidates_str
            )}
        ]
        # print('\nprompt: ', messages[1]["content"])
        
        result = await self.llm.call_llm(
            messages,
            self.model,
            temperature=self.temperature,  # Aggregate using the original low temperature
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        # Record the token statistics for the aggregator stage
        step_tokens["aggregator"]["input_tokens"] = result["input_tokens"]
        step_tokens["aggregator"]["output_tokens"] = result["output_tokens"]
        step_tokens["aggregator"]["total_tokens"] = result["total_tokens"]
        
        raw_output = result["response"]
        aggregated_sql = self.extractor.extract_sql(raw_output)
        # print('\nraw_output: ', raw_output)

        # refiner
        refiner = FeedbackBasedRefiner(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)

        refine_result = await refiner.process_sql(aggregated_sql, data_file, query_id)

        step_tokens["refine"]["input_tokens"] = refine_result["input_tokens"]
        step_tokens["refine"]["output_tokens"] = refine_result["output_tokens"]
        step_tokens["refine"]["total_tokens"] = refine_result["total_tokens"]

        refiner_raw_output = refine_result["response"]
        extracted_refined_sql = self.extractor.extract_sql(refiner_raw_output)


        # Calculate the total token
        total_tokens = {
            "input_tokens": sum(step["input_tokens"] for step in step_tokens.values()),
            "output_tokens": sum(step["output_tokens"] for step in step_tokens.values()),
            "total_tokens": sum(step["total_tokens"] for step in step_tokens.values())
        }
        # # Calculate the total token usage
        # total_tokens = {
        #     "input_tokens": step_tokens["candidates"]["input_tokens"] + step_tokens["aggregator"]["input_tokens"],
        #     "output_tokens": step_tokens["candidates"]["output_tokens"] + step_tokens["aggregator"]["output_tokens"],
        #     "total_tokens": step_tokens["candidates"]["total_tokens"] + step_tokens["aggregator"]["total_tokens"]
        # }
        
        # Save the intermediate result
        self.save_intermediate(
            input_data={
                "query": query,
                "formatted_schema": formatted_schema,
                "candidates": candidates
            },
            output_data={
                "raw_output": raw_output,
                "aggregated_sql": aggregated_sql,
                "extracted_sql": extracted_refined_sql,
                "all_candidates": candidates
            },
            model_info={
                "model": self.model,
                "input_tokens": total_tokens["input_tokens"],
                "output_tokens": total_tokens["output_tokens"],
                "total_tokens": total_tokens["total_tokens"],
                "step_tokens": step_tokens
            },
            query_id=query_id,
            module_name=self.name if (module_name == None) else module_name
        )
        
        self.log_io(
            input_data={
                "query": query,
                "formatted_schema": formatted_schema,
                "candidates": candidates
            },
            output_data=refiner_raw_output
        )
        
        return refiner_raw_output 