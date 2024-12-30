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

    def log_call(self, model: str, input_tokens: int, output_tokens: int):
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
        
    def _get_model_name(self, model_key: str) -> str:
        """获取实际的模型名称"""
        return self.config["models"][model_key]["model_name"]

    async def call_llm(self, 
                     messages: List[Dict[str, str]], 
                     model: str,
                     temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        调用LLM API
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "xxx"}, ...]
            model: 模型配置中的key（如"gpt4"或"gpt35"）
            temperature: 温度参数
            
        Returns:
            Dict[str, Any]: API响应结果
        """
        model_name = self._get_model_name(model)
        
        # 调用API
        response = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.config["models"][model]["temperature"],
            max_tokens=self.config["models"][model]["max_tokens"]
        )
        
        # 从response中获取token统计
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # 记录metrics
        self.metrics.log_call(model, input_tokens, output_tokens)
        
        return {
            "response": response.choices[0].message.content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": response.usage.total_tokens,
            "finish_reason": response.choices[0].finish_reason
        } 