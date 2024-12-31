from typing import Dict, Any, Optional, List
import yaml
import tiktoken
from openai import AsyncOpenAI
from abc import ABC, abstractmethod

class LLMMetrics:
    """用于记录LLM调用的指标"""
    def __init__(self):
        self.total_calls = 0
        self.model_calls = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.model_tokens = {}
        self.module_stats = {}  # 新增：按模块统计

    def log_call(self, model: str, input_tokens: int, output_tokens: int, module_name: str = None):
        """记录一次LLM调用"""
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        if model not in self.model_calls:
            self.model_calls[model] = 0
            self.model_tokens[model] = {"input": 0, "output": 0}
        
        self.model_calls[model] += 1
        self.model_tokens[model]["input"] += input_tokens
        self.model_tokens[model]["output"] += output_tokens
        
        # 按模块统计
        if module_name:
            if module_name not in self.module_stats:
                self.module_stats[module_name] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            self.module_stats[module_name]["calls"] += 1
            self.module_stats[module_name]["input_tokens"] += input_tokens
            self.module_stats[module_name]["output_tokens"] += output_tokens

class LLMBase:
    """LLM调用的基础类"""
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.client = AsyncOpenAI(
            api_key=self.config["api"]["api_key"],
            base_url=self.config["api"]["base_url"]
        )
        self.metrics = LLMMetrics()

    async def call_llm(self, 
                     messages: List[Dict[str, str]], 
                     model: str,
                     temperature: float = 0.0,
                     max_tokens: int = 1000,
                     module_name: str = None) -> Dict[str, Any]:
        """
        调用LLM API
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "xxx"}, ...]
            model: 模型名称（如"gpt-4"或"gpt-3.5-turbo"）
            temperature: 温度参数
            max_tokens: 最大token数
            module_name: 调用模块的名称，用于统计
            
        Returns:
            Dict[str, Any]: API响应结果
        """
        # 调用API
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 从response中获取token统计
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # 记录metrics
        self.metrics.log_call(model, input_tokens, output_tokens, module_name)
        
        return {
            "response": response.choices[0].message.content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": response.usage.total_tokens,
            "finish_reason": response.choices[0].finish_reason
        } 