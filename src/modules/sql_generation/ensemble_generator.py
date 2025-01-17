from typing import Dict, List, Optional
from .enhanced_generator import EnhancedSQLGenerator
from .submodules.selector import DirectSelector
from ...core.llm import LLMBase

class EnsembleGenerator(EnhancedSQLGenerator):
    """使用集成方法生成SQL的实现，基于EnhancedSQLGenerator生成多个候选并选择最佳结果，使用DirectSelector选择。故继承自EnhancedSQLGenerator"""
    
    def __init__(self, 
                llm: LLMBase, 
                model: str = "gpt-3.5-turbo-0613",
                temperature: float = 0.7,  # 使用更高的temperature以获得多样性
                max_tokens: int = 5000,
                n_candidates: int = 3,  # 生成的候选数量
                max_retries: int = 3):  # 重试次数阈值
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
            temperature=0.0,  # selector使用低temperature
            max_tokens=max_tokens
        )
        
    async def generate_sql(self, query: str, schema_linking_output: Dict, query_id: str, module_name: Optional[str] = None) -> str:
        """生成多个候选SQL并选择最佳结果"""
        # 记录每个步骤的token统计
        step_tokens = {
            "candidates": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "selector": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
        
        # 1. 生成多个候选SQL
        candidates = []
        for i in range(self.n_candidates):
            print(f"生成第 {i+1}/{self.n_candidates} 个候选SQL...")
            try:
                # 调用父类的generate_sql方法
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
                        
                        # 累加token统计
                        try:
                            prev_result = self.load_previous_result(query_id)
                            model_info = prev_result["model_info"]
                            step_tokens["candidates"]["input_tokens"] += model_info["input_tokens"]
                            step_tokens["candidates"]["output_tokens"] += model_info["output_tokens"]
                            step_tokens["candidates"]["total_tokens"] += model_info["total_tokens"]
                        except Exception as e:
                            self.logger.warning(f"无法加载候选{i}的token统计: {str(e)}")
                    else:
                        self.logger.warning(f"候选{i}的SQL提取失败")
                else:
                    self.logger.warning(f"候选{i}生成失败")
                    
            except Exception as e:
                self.logger.error(f"生成候选{i}时发生错误: {str(e)}")
                continue
        
        if not candidates:
            self.logger.error("没有成功生成任何候选SQL")
            raise ValueError("生成候选SQL失败")
            
        print(f"成功生成了 {len(candidates)} 个候选SQL")
        print(candidates)
        
        # 2. 使用selector选择最佳SQL
        print("开始选择最佳SQL...")
        try:
            data_file = self.data_file
            result = await self.selector.select_sql(data_file, candidates, query_id)
            selected_sql = result
            
            # 直接从selector的返回结果中获取token统计
            step_tokens["selector"] = {
                "input_tokens": 0,  # selector模块继承自ModuleBase，会自己记录token统计
                "output_tokens": 0,
                "total_tokens": 0
            }
                
        except Exception as e:
            self.logger.error(f"选择最佳SQL时发生错误: {str(e)}")
            # 如果选择失败，返回第一个候选
            selected_sql = candidates[0]
        
        # 计算总token
        total_tokens = {
            "input_tokens": sum(step["input_tokens"] for step in step_tokens.values()),
            "output_tokens": sum(step["output_tokens"] for step in step_tokens.values()),
            "total_tokens": sum(step["total_tokens"] for step in step_tokens.values())
        }
        
        # 保存最终的中间结果
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
        
        # 保持返回值格式与raw_output统一，使用```sql```包裹selected_sql
        return f"```sql\n{selected_sql}\n```"