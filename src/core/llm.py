from typing import Dict, Any, Optional, List
import yaml
import tiktoken
import torch
from openai import AsyncOpenAI
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio

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
        # 如果model为none，不记录统计信息
        if model.lower() == "none":
            return
            
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

class LocalModelHandler:
    """处理本地模型的加载和推理"""
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        
    def load_model(self, model_name):
        """加载指定的本地模型"""
        if model_name in self.models:
            return self.models[model_name], self.tokenizers[model_name]
            
        # 检查模型是否在配置中
        if model_name not in self.config["local_models"]:
            raise ValueError(f"本地模型 {model_name} 未在配置文件中定义")
            
        model_config = self.config["local_models"][model_name]
        model_path = model_config["path"]
        
        print(f"Loading local model {model_name} from {model_path}...")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side='right'
        )
        
        # 设置padding token，确保它与eos_token不同
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 如果添加了新的special token，需要调整模型的embedding
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        
        # 缓存模型和tokenizer
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        return model, tokenizer
        
    def _clean_response(self, text: str) -> str:
        """清理模型输出中的特殊标记"""
        # 清理Qwen特有的标记
        special_tokens = [
            "<|im_start|>", "<|im_end|>",  # 对话标记
            "<|assistant|>", "<|user|>", "<|system|>",  # 角色标记
            "<|repo_name|>",  # 仓库标记
            "<|commit_hash|>",  # 提交标记
            "<|path|>",  # 路径标记
            "<|diff|>",  # diff标记
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",  # FIM相关标记
            "<|endoftext|>",  # 文本结束标记
        ]
        
        cleaned_text = text
        for token in special_tokens:
            cleaned_text = cleaned_text.replace(token, "")
            
        # 清理连续的空行
        cleaned_text = "\n".join(
            line for line in cleaned_text.splitlines() 
            if line.strip()
        )
        
        # 清理首尾空白
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
        
    async def generate(self, model_name, messages, temperature=0.0, max_tokens=1000):
        """使用本地模型生成回复"""
        model, tokenizer = self.load_model(model_name)
        
        # 将消息格式转换为模型输入
        prompt = self._convert_messages_to_prompt(model_name, messages, tokenizer)
        
        # 编码输入，显式设置padding和attention mask
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # 可以根据需要调整
            return_attention_mask=True
        )
        
        # 将输入移动到正确的设备
        input_ids = encoded['input_ids'].to(model.device)
        attention_mask = encoded['attention_mask'].to(model.device)
        input_tokens = len(input_ids[0])
        
        # 使用模型生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # 显式传入attention mask
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
        # 解码输出并清理特殊标记
        raw_output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        cleaned_output = self._clean_response(raw_output)
        output_tokens = len(outputs[0]) - input_ids.shape[1]
        
        return {
            "response": cleaned_output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "finish_reason": "stop"
        }
        
    def _convert_messages_to_prompt(self, model_name, messages, tokenizer):
        """将OpenAI格式的消息转换为本地模型的提示格式"""
        model_config = self.config["local_models"][model_name]
        prompt_format = model_config.get("prompt_format", "default")
        
        if prompt_format == "qwen":
            # Qwen系列模型的提示格式
            prompt = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
            # 添加最后的assistant标记，等待模型生成
            prompt += "<|im_start|>assistant\n"
            return prompt
        else:
            # 默认格式，简单拼接
            prompt = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
                
            prompt += "Assistant: "
            return prompt

class LLMBase:
    """LLM调用的基础类"""
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=self.config["api"]["api_key"],
            base_url=self.config["api"]["base_url"]
        )
        
        # 初始化本地模型处理器
        self.local_handler = LocalModelHandler(self.config)
        
        # 初始化指标记录
        self.metrics = LLMMetrics()

    async def call_llm(self, 
                     messages: List[Dict[str, str]], 
                     model: str,
                     temperature: float = 0.0,
                     max_tokens: int = 1000,
                     module_name: str = None) -> Dict[str, Any]:
        """
        调用LLM API或本地模型
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "xxx"}, ...]
            model: 模型名称，如"gpt-4"或"qwen2.5-coder-3B"
            temperature: 温度参数
            max_tokens: 最大token数
            module_name: 调用模块的名称，用于统计
            
        Returns:
            Dict[str, Any]: API响应结果
        """
        # 检查是否使用本地模型
        if model in self.config.get("local_models", {}):
            # 使用本地模型
            response_data = await self.local_handler.generate(
                model_name=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # 使用OpenAI API
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_data = {
                "response": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason
            }
        
        # 记录metrics
        self.metrics.log_call(
            model, 
            response_data["input_tokens"], 
            response_data["output_tokens"], 
            module_name
        )
        
        return response_data 