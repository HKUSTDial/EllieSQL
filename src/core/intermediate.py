import json
import os
from typing import Dict, Any, List
from datetime import datetime

class IntermediateResult:
    """处理模块中间结果的基类"""
    
    def __init__(self, module_name: str, pipeline_id: str = None):
        """
        初始化中间结果处理器
        
        Args:
            module_name: 模块名称
            pipeline_id: pipeline的唯一标识，如果为None则使用时间戳
        """
        self.module_name = module_name
        self.pipeline_id = pipeline_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建目录结构
        self.base_dir = "results/intermediate_results"
        self.pipeline_dir = os.path.join(self.base_dir, self.pipeline_id)
        os.makedirs(self.pipeline_dir, exist_ok=True)
        
        # 中间结果文件路径
        self.result_file = os.path.join(self.pipeline_dir, f"{module_name}.jsonl")
        
        # Pipeline统计文件
        self.stats_file = os.path.join(self.pipeline_dir, "api_stats.json")
        
        # SQL结果文件
        self.sql_results_file = os.path.join(self.pipeline_dir, "generated_sql_results.jsonl")
        
    def save_result(self, 
                   input_data: Dict[str, Any],
                   output_data: Dict[str, Any],
                   model_info: Dict[str, Any],
                   query_id: str = None,
                   module_name: str = None) -> str:
        """
        保存中间结果和更新统计信息
        
        Args:
            input_data: 输入数据
            output_data: 输出数据
            model_info: 模型信息
            query_id: 查询ID
            module_name: 可选的模块名称，如果不指定则使用默认模块名
        
        Returns:
            str: 保存的文件路径
        """
        # 使用指定的模块名或默认模块名
        save_name = module_name if module_name else self.module_name
        
        # 中间结果文件路径
        result_file = os.path.join(self.pipeline_dir, f"{save_name}.jsonl")
        
        # 保存中间结果
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query_id": query_id or "unknown",
            "input": input_data,
            "output": output_data,
            "model_info": model_info
        }
        
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
        # 更新统计信息
        self._update_stats(model_info)
        
        return result_file
        
    def load_previous_result(self, 
                           query_id: str,
                           previous_module: str) -> Dict[str, Any]:
        """加载上一个模块的结果"""
        prev_file = os.path.join(self.pipeline_dir, f"{previous_module}.jsonl")
        if not os.path.exists(prev_file):
            raise FileNotFoundError(f"找不到上一个模块的结果文件: {prev_file}")
            
        with open(prev_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"结果文件为空: {prev_file}")
            # 返回最后一条结果
            return json.loads(lines[-1])
            
    def _update_stats(self, model_info: Dict[str, Any]):
        """更新API调用统计信息"""
        # 如果model为none，不更新统计信息
        if model_info.get("model", "").lower() == "none":
            return
        
        stats = self._load_json(self.stats_file, {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_calls": 0,
            "total_cost": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            "models": {},
            "modules": {}
        })
        
        # 更新总调用次数和总消耗
        stats["total_calls"] += 1
        stats["total_cost"]["input_tokens"] += model_info["input_tokens"]
        stats["total_cost"]["output_tokens"] += model_info["output_tokens"]
        stats["total_cost"]["total_tokens"] += (model_info["input_tokens"] + 
                                              model_info["output_tokens"])
        
        # 更新模型统计
        model_name = model_info["model"]
        if model_name not in stats["models"]:
            stats["models"][model_name] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
            
        model_stats = stats["models"][model_name]
        model_stats["calls"] += 1
        model_stats["input_tokens"] += model_info["input_tokens"]
        model_stats["output_tokens"] += model_info["output_tokens"]
        model_stats["total_tokens"] += model_info["input_tokens"] + model_info["output_tokens"]
        
        # 更新模块统计
        if self.module_name not in stats["modules"]:
            stats["modules"][self.module_name] = {
                "calls": 0,
                "models": {}
            }
            
        module_stats = stats["modules"][self.module_name]
        module_stats["calls"] += 1
        
        if model_name not in module_stats["models"]:
            module_stats["models"][model_name] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
            
        model_module_stats = module_stats["models"][model_name]
        model_module_stats["calls"] += 1
        model_module_stats["input_tokens"] += model_info["input_tokens"]
        model_module_stats["output_tokens"] += model_info["output_tokens"]
        model_module_stats["total_tokens"] += (model_info["input_tokens"] + 
                                             model_info["output_tokens"])
        
        # 更新最后修改时间
        stats["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_json(self.stats_file, stats)
        
    def _load_json(self, file_path: str, default: Dict) -> Dict:
        """加载JSON文件，如果文件不存在则返回默认值"""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default
        
    def _save_json(self, file_path: str, data: Dict):
        """保存JSON文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2) 
        
    def save_sql_result(self, query_id: str, source: str, sql: str):
        """保存SQL结果到JSONL文件"""
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question_id": query_id,
            "source": source,
            "generated_sql": sql
        }
        
        with open(self.sql_results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n") 