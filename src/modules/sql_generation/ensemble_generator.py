from typing import Dict, List, Optional
from .enhanced_generator import EnhancedSQLGenerator
from .submodules.selector import DirectSelector
from ...core.llm import LLMBase

class EnsembleGenerator(EnhancedSQLGenerator):
    """Implementation of using ensemble method to generate SQL, based on EnhancedSQLGenerator to generate multiple candidates and select the best result, using DirectSelector to select. Therefore, it inherits from EnhancedSQLGenerator"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.7,  # Use higher temperature to get diversity
                max_tokens: int = 5000,
                n_candidates: int = 3,  # The number of candidates generated
                max_retries: int = 3):  # The threshold for the number of retries
        super().__init__(
            llm=llm,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        )
        self.name = "EnsembleGenerator"
        self.n_candidates = n_candidates
        self.selector = DirectSelector(
            llm=llm,
            model=model,
            temperature=0.0,  # selector uses low temperature
            max_tokens=max_tokens
        )
        
    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str, module_name: Optional[str] = None) -> str:
        """Generate multiple candidate SQLs and select the best result"""
        # Record the token statistics for each step
        step_tokens = {
            "candidates": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "selector": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
        
        # 1. Generate multiple candidate SQLs
        candidates = []
        for i in range(self.n_candidates):
            print(f"Generating the {i+1}/{self.n_candidates} candidate SQL...")
            try:
                # Call the generate_sql method of the parent class
                candidate_result = await super().generate_sql(
                    query=query,
                    schema_linking_output=schema_linking_output,
                    query_id=query_id,
                    module_name=self.name
                )
                
                if candidate_result:
                    extracted_sql = self.extractor.extract_sql(candidate_result)
                    if extracted_sql:
                        candidates.append(extracted_sql)
                        
                        # Add token statistics
                        try:
                            prev_result = self.load_previous_result(query_id)
                            model_info = prev_result["model_info"]
                            step_tokens["candidates"]["input_tokens"] += model_info["input_tokens"]
                            step_tokens["candidates"]["output_tokens"] += model_info["output_tokens"]
                            step_tokens["candidates"]["total_tokens"] += model_info["total_tokens"]
                        except Exception as e:
                            self.logger.warning(f"Failed to load the token statistics for the {i} candidate: {str(e)}")
                    else:
                        self.logger.warning(f"Failed to extract the SQL for the {i} candidate")
                else:
                    self.logger.warning(f"Failed to generate the {i} candidate")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate the {i} candidate: {str(e)}")
                continue
        
        if not candidates:
            self.logger.error("Failed to generate any candidate SQL")
            raise ValueError("Failed to generate candidate SQL")
            
        print(f"Successfully generated {len(candidates)} candidate SQLs")
        print(candidates)
        
        # 2. Use selector to select the best SQL
        print("Start selecting the best SQL...")
        try:
            data_file = self.data_file
            result = await self.selector.select_sql(data_file, candidates, query_id)
            selected_sql = result
            
            # Get token statistics from the return result of selector
            step_tokens["selector"] = {
                "input_tokens": 0,  # selector module inherits from ModuleBase, will record token statistics by itself
                "output_tokens": 0,
                "total_tokens": 0
            }
                
        except Exception as e:
            self.logger.error(f"Failed to select the best SQL: {str(e)}")
            # If selection fails, return the first candidate
            selected_sql = candidates[0]
        
        # Calculate the total token
        total_tokens = {
            "input_tokens": sum(step["input_tokens"] for step in step_tokens.values()),
            "output_tokens": sum(step["output_tokens"] for step in step_tokens.values()),
            "total_tokens": sum(step["total_tokens"] for step in step_tokens.values())
        }
        
        # Save the final intermediate result
        self.save_intermediate(
            input_data={
                "query": query,
                "schema_linking_output": schema_linking_output,
                "candidates": candidates
            },
            output_data={
                "selected_sql": selected_sql,
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
        
        # Keep the return value format consistent with raw_output, wrap selected_sql with ```sql```
        return f"```sql\n{selected_sql}\n```"