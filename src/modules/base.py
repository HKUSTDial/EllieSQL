from typing import Any, Dict, Optional
from ..core.intermediate import IntermediateResult


class ModuleBase:
    """所有模块的基类"""
    def __init__(self, name: str):
        self.name = name
        self.intermediate = IntermediateResult(name)
        
    def log_io(self, input_data: Any, output_data: Any):
        """打印模块的输入输出"""
        print(f"\n{'-'*50}")
        print(f"Module: {self.name}")
        print(f"Input: {input_data}")
        print(f"Output: {output_data}")
        print(f"{'-'*50}\n")
        
    def save_intermediate(self, 
                        input_data: Dict[str, Any],
                        output_data: Dict[str, Any],
                        model_info: Dict[str, Any],
                        query_id: Optional[str] = None) -> str:
        """保存中间结果"""
        return self.intermediate.save_result(
            input_data=input_data,
            output_data=output_data,
            model_info=model_info,
            query_id=query_id
        )
        
    def load_previous(self, 
                     query_id: str,
                     previous_module: str) -> Dict[str, Any]:
        """加载上一个模块的结果"""
        return self.intermediate.load_previous_result(
            query_id=query_id,
            previous_module=previous_module
        ) 