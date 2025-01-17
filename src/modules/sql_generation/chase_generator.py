from typing import Dict, List, Optional
from .base import SQLGeneratorBase
from ...core.llm import LLMBase
from .prompts.gpt_prompts import SQL_GENERATION_SYSTEM, SQL_GENERATION_USER
from ...core.utils import load_json

from .submodules.refiner import *
from .submodules.divide_conqueror import *
from .submodules.query_planer import *
from .submodules.online_synthesiser import *
from .submodules.selector import *

class CHASESQLGenerator(SQLGeneratorBase):
    """使用CHASE pipeline生成SQL的实现"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.0,
                max_tokens: int = 5000,
                max_retries: int = 3):
        super().__init__("CHASESQLGenerator", max_retries)
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
            "divide_conquer": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "query_plan": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "online_synthesis": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "refine": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "select": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }

        # 分别使用DCR、QPR、OSR生成候选sql
        ssql = []

        #DCR
        divide_conqueror = DivideConqueror(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)

        result = await divide_conqueror.generate_sql(query=query, formatted_schema=formatted_schema, curr_evidence=curr_evidence)
        
        step_tokens["divide_conquer"]["input_tokens"] = result["input_tokens"]
        step_tokens["divide_conquer"]["output_tokens"] = result["output_tokens"]
        step_tokens["divide_conquer"]["total_tokens"] = result["total_tokens"]
        
        raw_output = result["response"]
        extracted_sql = self.extractor.extract_sql(raw_output)

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
        print("DC Refine完成")
        ssql.append(extracted_sql)
###################################################################################
        #QPR
        query_planer = QueryPlaner(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)
        result = await query_planer.generate_sql(query=query, formatted_schema=formatted_schema, curr_evidence=curr_evidence)
        
        step_tokens["query_plan"]["input_tokens"] = result["input_tokens"]
        step_tokens["query_plan"]["output_tokens"] = result["output_tokens"]
        step_tokens["query_plan"]["total_tokens"] = result["total_tokens"]
        
        raw_output = result["response"]
        extracted_sql = self.extractor.extract_sql(raw_output)

                #Refine阶段
        refiner = FeedbackBasedRefiner(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)

        refine_result = await refiner.process_sql(extracted_sql, data_file, query_id)

        step_tokens["refine"]["input_tokens"] += refine_result["input_tokens"]
        step_tokens["refine"]["output_tokens"] += refine_result["output_tokens"]
        step_tokens["refine"]["total_tokens"] += refine_result["total_tokens"]

        refiner_raw_output = refine_result["response"]
        extracted_sql = self.extractor.extract_sql(refiner_raw_output)
        print("QP Refine完成")
        ssql.append(extracted_sql)
###################################################################################
        #OSR
        online_synthesisor = OnlineSynthesiser(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)
        result = await online_synthesisor.generate_sql(query=query, formatted_schema=formatted_schema, curr_evidence=curr_evidence)
        
        step_tokens["online_synthesis"]["input_tokens"] = result["input_tokens"]
        step_tokens["online_synthesis"]["output_tokens"] = result["output_tokens"]
        step_tokens["online_synthesis"]["total_tokens"] = result["total_tokens"]
        
        raw_output = result["response"]
        extracted_sql = self.extractor.extract_sql(raw_output)

                #Refine阶段
        refiner = FeedbackBasedRefiner(llm=self.llm, 
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                module_name=self.name)

        refine_result = await refiner.process_sql(extracted_sql, data_file, query_id)

        step_tokens["refine"]["input_tokens"] += refine_result["input_tokens"]
        step_tokens["refine"]["output_tokens"] += refine_result["output_tokens"]
        step_tokens["refine"]["total_tokens"] += refine_result["total_tokens"]

        refiner_raw_output = refine_result["response"]
        extracted_sql = self.extractor.extract_sql(refiner_raw_output)
        print("OS Refine完成")
        ssql.append(extracted_sql)


        ## Select 阶段, 使用selector选择最佳SQL
        selector = DirectSelector(
            llm=self.llm,
            model=self.model,
            temperature=0.0,  # selector使用低temperature
            max_tokens=self.max_tokens
        )

        print(f"成功生成了 {len(ssql)} 个候选SQL")
        # print(ssql)

        print("开始选择最佳SQL...")
        selected_sql = ""
        try:
            data_file = self.data_file
            result = await selector.select_sql(data_file, ssql, query_id)

            step_tokens["select"]["input_tokens"] = result["input_tokens"]
            step_tokens["select"]["output_tokens"] = result["output_tokens"]
            step_tokens["select"]["total_tokens"] = result["total_tokens"]

            raw_output = result["response"]
                
            selected_sql = self.extractor.extract_sql(raw_output)

        except Exception as e:
            self.logger.error(f"选择最佳SQL时发生错误: {str(e)}")
            # 如果选择失败，返回第一个候选
            selected_sql = ssql[0]

        
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
                # "messages": "mm" #messages
            },
            output_data={
                "raw_output": raw_output,
                # "extracted_sql": extracted_sql
                "candidates": ssql,
                "selected_sql": selected_sql,
                "extracted_sql": selected_sql
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
                "messages": "mm" #messages
            }, 
            output_data=raw_output
        )
        
        return selected_sql 
        