import json
import os
from typing import Dict, Any, List
from datetime import datetime

class IntermediateResult:
    """处理模块中间结果的基类"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.results_dir = f"intermediate_results/{module_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def save_result(self, 
                   input_data: Dict[str, Any],
                   output_data: Dict[str, Any],
                   model_info: Dict[str, Any],
                   query_id: str = None) -> str:
        """
        保存中间结果
        
        Args:
            input_data: 输入数据
            output_data: 输出数据
            model_info: 模型调用信息，包含:
                {
                    "model": "模型名称",
                    "input_tokens": 输入token数,
                    "output_tokens": 输出token数,
                    "total_tokens": 总token数
                }
            query_id: 查询ID，如果为None则使用时间戳
            
        Returns:
            str: 结果文件路径
        """
        if query_id is None:
            query_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        result = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "output": output_data,
            "model_info": model_info
        }
        
        filepath = os.path.join(self.results_dir, f"{query_id}.jsonl")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
        return filepath
        
    def load_previous_result(self, 
                           query_id: str,
                           previous_module: str) -> Dict[str, Any]:
        """
        加载上一个模块的结果
        
        Args:
            query_id: 查询ID
            previous_module: 上一个模块的名称
            
        Returns:
            Dict[str, Any]: 上一个模块的最后一条结果
        """
        filepath = f"intermediate_results/{previous_module}/{query_id}.jsonl"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到上一个模块的结果文件: {filepath}")
            
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"结果文件为空: {filepath}")
            # 返回最后一条结果
            return json.loads(lines[-1]) 