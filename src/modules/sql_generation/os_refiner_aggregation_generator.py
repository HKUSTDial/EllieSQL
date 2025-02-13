from typing import Dict, List, Optional
from .online_synthesis_refiner_generator import OSRefinerSQLGenerator
from ...core.llm import LLMBase
from .prompts.aggregator_prompts import AGGREGATOR_SYSTEM, AGGREGATOR_USER
from ...core.utils import load_json
from .submodules.refiner import *

class OSRefinerAggregationSQLGenerator(OSRefinerSQLGenerator):
    """使用多个OSRefiner生成候选并聚合的SQL生成器，继承自OSRefinerSQLGenerator"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,  # 聚合时使用的temperature
                max_tokens: int = 5000,
                n_candidates: int = 3,  # 生成的候选数量
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
        """生成多个候选SQL并聚合"""
        # 记录每个步骤的token统计
        step_tokens = {
            "candidates": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "aggregator": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "refine": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
        
        # 加载schema linking的结果
        if not schema_linking_output:
            prev_result = self.load_previous_result(query_id)
            formatted_schema = prev_result["output"]["formatted_linked_schema"]
        else:
            formatted_schema = schema_linking_output.get("formatted_linked_schema")
            if not formatted_schema:
                prev_result = self.load_previous_result(query_id)
                formatted_schema = prev_result["output"]["formatted_linked_schema"]
                
        # 获取evidence
        data_file = self.data_file
        dataset_examples = load_json(data_file)
        curr_evidence = ""
        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                curr_evidence = item.get("evidence", "")
                break
                
        # 1. 生成多个候选SQL
        candidates = []
        candidates_gen_temperature = 1.0  # 生成候选SQL时使用较高的temperature以获得多样性
        
        # 保存原始temperature
        original_temperature = self.temperature
        
        try:
            # 临时修改temperature为高温
            self.temperature = candidates_gen_temperature
            
            for i in range(self.n_candidates):
                # 直接调用父类的generate_sql方法生成候选
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
                    
                # 累加candidates阶段的token统计
                if hasattr(self, '_last_result'):
                    step_tokens["candidates"]["input_tokens"] += self._last_result["input_tokens"]
                    step_tokens["candidates"]["output_tokens"] += self._last_result["output_tokens"]
                    step_tokens["candidates"]["total_tokens"] += self._last_result["total_tokens"]
                    
        finally:
            # 恢复原始temperature
            self.temperature = original_temperature
                
        if not candidates:
            self.logger.error(f"未能生成有效的候选SQL! Question ID: {query_id}")
            return "```sql\nSELECT 1;\n```"  # 返回一个简单的fallback SQL
            
        # 2. 聚合候选SQL (使用原始的低temperature)
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
            temperature=self.temperature,  # 聚合使用原始的低temperature
            max_tokens=self.max_tokens,
            module_name=self.name
        )
        
        # 记录aggregator阶段的token统计
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


        # 计算总token
        total_tokens = {
            "input_tokens": sum(step["input_tokens"] for step in step_tokens.values()),
            "output_tokens": sum(step["output_tokens"] for step in step_tokens.values()),
            "total_tokens": sum(step["total_tokens"] for step in step_tokens.values())
        }
        # # 计算总token使用量
        # total_tokens = {
        #     "input_tokens": step_tokens["candidates"]["input_tokens"] + step_tokens["aggregator"]["input_tokens"],
        #     "output_tokens": step_tokens["candidates"]["output_tokens"] + step_tokens["aggregator"]["output_tokens"],
        #     "total_tokens": step_tokens["candidates"]["total_tokens"] + step_tokens["aggregator"]["total_tokens"]
        # }
        
        # 保存中间结果
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