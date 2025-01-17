from typing import Dict, List, Optional
from .base import SQLGeneratorBase
from ...core.llm import LLMBase
from .prompts.query_plan_cot_prompts import SQL_GENERATION_SYSTEM, QUERY_PLAN_PROMPT
from .submodules.refiner import *
from .submodules.query_planer import *
from ...core.utils import load_json

class QPRefinerSQLGenerator(SQLGeneratorBase):
    """使用Query Plan + Refiner生成SQL的实现"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 5000,
                max_retries: int = 3):
        super().__init__("QPRefinerSQLGenerator", max_retries)
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str, module_name: Optional[str] = None) -> str:
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
        
        data_file = self.data_file
        dataset_examples = load_json(data_file)
        curr_evidence = ""
        for item in dataset_examples:
            if(item.get("question_id") == query_id):
                curr_evidence = item.get("evidence", "")
                break

                # 记录每个步骤的token统计
        step_tokens = {
            "query_plan": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "refine": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }


        query_planer = QueryPlaner(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)
        result = await query_planer.generate_sql(query=query, formatted_schema=formatted_schema, curr_evidence=curr_evidence)
        
        step_tokens["query_plan"]["input_tokens"] = result["input_tokens"]
        step_tokens["query_plan"]["output_tokens"] = result["output_tokens"]
        step_tokens["query_plan"]["total_tokens"] = result["total_tokens"]
        
        raw_output_before_refine = result["response"]
        # print('-------------')
        # print(raw_output_before_refine)
        extracted_sql = self.extractor.extract_sql(raw_output_before_refine)

        # # 构建提示词
        # messages = [
        #     {"role": "system", "content": SQL_GENERATION_SYSTEM},
        #     {"role": "user", "content": QUERY_PLAN_PROMPT.format(
        #         schema=formatted_schema,
        #         query=query,
        #         evidence=curr_evidence if curr_evidence else "None"
        #     )}
        # ]
        
        # # print('\n'+messages[1]['content']+'\n')
        # result = await self.llm.call_llm(
        #     messages,
        #     self.model,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     module_name=self.name
        # )
        
        # raw_output = result["response"]
        # extracted_sql = self.extractor.extract_sql(raw_output)

                #Refine阶段
        refiner = FeedbackBasedRefiner(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)


        refine_result = await refiner.process_sql(extracted_sql, data_file, query_id)

        step_tokens["refine"]["input_tokens"] = refine_result["input_tokens"]
        step_tokens["refine"]["output_tokens"] = refine_result["output_tokens"]
        step_tokens["refine"]["total_tokens"] = refine_result["total_tokens"]


        refiner_raw_output = refine_result["response"]
        extracted_sql = self.extractor.extract_sql(refiner_raw_output)


        # 计算总token
        total_tokens = {
            "input_tokens": sum(step["input_tokens"] for step in step_tokens.values()),
            "output_tokens": sum(step["output_tokens"] for step in step_tokens.values()),
            "total_tokens": sum(step["total_tokens"] for step in step_tokens.values())
        }
        
        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query,
                "formatted_schema": formatted_schema,
                # "messages": "mm"#messages
            },
            output_data={
                "raw_output": refiner_raw_output,
                "extracted_sql": extracted_sql
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
                # "messages": "mm"#messages
            }, 
            output_data=refiner_raw_output
        )
        
        return refiner_raw_output
        