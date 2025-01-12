from ..src.modules.sql_generation.enhanced_generator import *
from ..src.modules.sql_generation.submodules.selector import *

class EnsembleSQLGenerator(SQLGeneratorBase):
    """通过组合数条Divide And Conquer CoT + Online synthesis SQL generator, 通过select选出最终的sql"""
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.5,
                max_tokens: int = 10000):
        super().__init__(name="EnsembleSQLGenerator")
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



        candidate_num = 3

        # 将候选sql保存在列表ssql中
        ssql = []
        
        for i in range(candidate_num):
            enhanced_SQL_generator = EnhancedSQLGenerator(llm = self.llm, model = self.model, temperature = self.temperature, max_tokens = self.max_tokens)
            # print(f"实例化enhanced_SQL_generator 成功{i}")
            candidate_sql = await enhanced_SQL_generator.generate_sql2(query = query, formatted_schema = formatted_schema, query_id = query_id)
            # print(f"candidate_sql 生成成功: {candidate_sql}" )
            ssql.append(candidate_sql)

            print(len(ssql))


        print("能跳出循环")
        director_selector = DirectSelector(llm = self.llm, model = self.model, temperature = self.temperature, max_tokens = self.max_tokens)
        print("DirectSelector实例化成功")
        extracted_sql = await director_selector.select_sql(query, ssql, query_id)

        print("sql selector 完成")





        # 保存中间结果
        self.save_intermediate(
            input_data={
                "query": query,
                "formatted_schema": formatted_schema
                # "messages": assemble_prompt #这样修改合理吗
            },
            output_data={
                "raw_output": ssql,
                "extracted_sql": extracted_sql
            },
            model_info={
                "model": self.model,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            query_id=query_id
        )
        
        self.log_io(
            input_data={
                "query": query, 
                "formatted_schema": formatted_schema,
                "messages": "" #这样修改合理吗
            }, 
            output_data= {
                "candidates": ssql,
                "final_sql": extracted_sql
            }
        )
        
        return extracted_sql
