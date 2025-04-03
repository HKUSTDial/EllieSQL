import json
import os
from typing import Dict, Any, List
from datetime import datetime
from .config import Config

class IntermediateResult:
    """Base class for processing module intermediate results"""
    
    def __init__(self, module_name: str, pipeline_id: str = None):
        """
        Initialize the intermediate result processor
        :param module_name: module name
        :param pipeline_id: unique identifier of the pipeline, if None then use timestamp
        """
        self.module_name = module_name
        self.pipeline_id = pipeline_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use the path configured in the config
        self.base_dir = Config().intermediate_results_dir
        self.pipeline_dir = self.base_dir / self.pipeline_id
        os.makedirs(self.pipeline_dir, exist_ok=True)
        
        # Intermediate result file path
        self.result_file = self.pipeline_dir / f"{module_name}.jsonl"
        
        # Pipeline statistics file
        self.stats_file = self.pipeline_dir / "api_stats.json"
        
        # SQL results file
        self.sql_results_file = self.pipeline_dir / "generated_sql_results.jsonl"
        
    def save_result(self, 
                   input_data: Dict[str, Any],
                   output_data: Dict[str, Any],
                   model_info: Dict[str, Any],
                   query_id: str = None,
                   module_name: str = None) -> str:
        """
        Save intermediate results and update statistics
        
        :param input_data: input data
        :param output_data: output data
        :param model_info: model information
        :param query_id: query ID
        :param module_name: optional module name, if not specified then use the default module name
        
        :return: saved file path
        """
        # Use the specified module name or the default module name
        save_name = module_name if module_name else self.module_name
        
        # Intermediate result file path
        result_file = os.path.join(self.pipeline_dir, f"{save_name}.jsonl")
        
        # Save intermediate results
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query_id": query_id or "unknown",
            "input": input_data,
            "output": output_data,
            "model_info": model_info
        }
        
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
        # Update statistics
        self._update_stats(model_info)
        
        return result_file
        
    def load_previous_result(self, 
                           query_id: str,
                           previous_module: str) -> Dict[str, Any]:
        """Load the result of the previous module"""
        prev_file = os.path.join(self.pipeline_dir, f"{previous_module}.jsonl")
        if not os.path.exists(prev_file):
            raise FileNotFoundError(f"找不到上一个模块的结果文件: {prev_file}")
            
        with open(prev_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"结果文件为空: {prev_file}")
            # Return the last result
            return json.loads(lines[-1])
            
    def _update_stats(self, model_info: Dict[str, Any]):
        """Update API call statistics"""
        # If model is none, do not update statistics
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
        
        # Update total calls and total cost
        stats["total_calls"] += 1
        stats["total_cost"]["input_tokens"] += model_info["input_tokens"]
        stats["total_cost"]["output_tokens"] += model_info["output_tokens"]
        stats["total_cost"]["total_tokens"] += (model_info["input_tokens"] + 
                                              model_info["output_tokens"])
        
        # Update model statistics
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
        
        # Update module statistics
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
        
        # Update the last modified time
        stats["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_json(self.stats_file, stats)
        
    def _load_json(self, file_path: str, default: Dict) -> Dict:
        """Load JSON file, return default value if file does not exist"""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default
        
    def _save_json(self, file_path: str, data: Dict):
        """Save JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2) 
        
    def save_sql_result(self, query_id: str, source: str, sql: str):
        """Save SQL results to JSONL file"""
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question_id": query_id,
            "source": source,
            "generated_sql": sql
        }
        
        with open(self.sql_results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n") 