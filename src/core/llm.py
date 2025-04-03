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
        self.module_stats = {}

    def log_call(self, model: str, input_tokens: int, output_tokens: int, module_name: str = None):
        """
        Log a LLM call
        :param model: model name
        :param input_tokens: input tokens
        :param output_tokens: output tokens
        :param module_name: module name
        """
        # If model is none, do not record statistics
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
        
        # Statistics by module
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
    """Deal with the loading and inference of local models"""
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        
    def load_model(self, model_name):
        """Load the specified local model"""
        if model_name in self.models:
            return self.models[model_name], self.tokenizers[model_name]
            
        # Check if the model is defined in the config
        if model_name not in self.config["local_models"]:
            raise ValueError(f"Local model {model_name} is not defined in the config file")
            
        model_config = self.config["local_models"][model_name]
        model_path = model_config["path"]
        
        print(f"Loading local model {model_name} from {model_path}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side='right'
        )
        
        # Set padding token, ensure it is different from eos_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # If new special tokens are added, adjust the model's embedding
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        
        # Cache model and tokenizer
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        return model, tokenizer
        
    def _clean_response(self, text: str) -> str:
        """Clean special tokens in the model output"""
        # Clean Qwen-specific tokens
        special_tokens = [
            "<|im_start|>", "<|im_end|>",  # Dialog tokens
            "<|assistant|>", "<|user|>", "<|system|>",  # Role tokens
            "<|repo_name|>",  # Repository tokens
            "<|commit_hash|>",  # Commit tokens
            "<|path|>",  # Path tokens
            "<|diff|>",  # Diff tokens
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",  # FIM related tokens
            "<|endoftext|>",  # Text end tokens
        ]
        
        cleaned_text = text
        for token in special_tokens:
            cleaned_text = cleaned_text.replace(token, "")
            
        # Clean consecutive empty lines
        cleaned_text = "\n".join(
            line for line in cleaned_text.splitlines() 
            if line.strip()
        )
        
        # Clean leading and trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
        
    async def generate(self, model_name, messages, temperature=0.0, max_tokens=1000):
        """Generate a response using a local model"""
        model, tokenizer = self.load_model(model_name)
        
        # Convert messages to model input
        prompt = self._convert_messages_to_prompt(model_name, messages, tokenizer)
        
        # Encode input, explicitly set padding and attention mask
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        )
        
        # Move input to the correct device
        input_ids = encoded['input_ids'].to(model.device)
        attention_mask = encoded['attention_mask'].to(model.device)
        input_tokens = len(input_ids[0])
        
        # Generate with model
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=0.8, # According to the report
                top_k=20,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
        # Decode output and clean special tokens
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
        """Convert OpenAI-format messages to local model prompt format"""
        model_config = self.config["local_models"][model_name]
        prompt_format = model_config.get("prompt_format", "default")
        
        if prompt_format == "qwen":
            # Qwen series model prompt format
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
                
            # Add the final assistant marker, waiting for model generation
            prompt += "<|im_start|>assistant\n"
            return prompt
        else:
            # Default format, simple concatenation
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
    """Base class for LLM calls"""
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.config["api"]["api_key"],
            base_url=self.config["api"]["base_url"]
        )
        
        # Initialize local model processor
        self.local_handler = LocalModelHandler(self.config)
        
        # Initialize metrics recording
        self.metrics = LLMMetrics()

    async def call_llm(self, 
                     messages: List[Dict[str, str]], 
                     model: str,
                     temperature: float = 0.0,
                     max_tokens: int = 1000,
                     module_name: str = None) -> Dict[str, Any]:
        """
        Call LLM API or local model
        
        :param messages: Message list, format: [{"role": "user", "content": "xxx"}, ...]
        :param model: Model name, e.g. "gpt-4" or "qwen2.5-coder-3B"
        :param temperature: Temperature parameter
        :param max_tokens: Maximum token number
        :param module_name: Name of the calling module, for statistics
            
        :return: API response result
        """
        # Check if local model is used
        if model in self.config.get("local_models", {}):
            # Use local model
            response_data = await self.local_handler.generate(
                model_name=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # Use API
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
        
        # Record metrics
        self.metrics.log_call(
            model, 
            response_data["input_tokens"], 
            response_data["output_tokens"], 
            module_name
        )
        
        return response_data 