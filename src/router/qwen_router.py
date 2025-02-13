from typing import Dict
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..sft.qwen_classifier_sft import QwenForSequenceClassification
from ..sft.instruction_templates import PipelineClassificationTemplates
import torch.nn.functional as F
from pathlib import Path

class QwenRouter(RouterBase):
    """基于微调的Qwen模型进行路由的路由器，支持分类头和生成式两种模式"""
    
    def __init__(self, name: str = "QwenRouter", classifier_head: bool = True, lora_path: str = None, seed: int = 42):
        super().__init__(name)
        self.config = Config()
        self.model_path = self.config.model_dir
        self.classifier_head = classifier_head
        
        # 使用指定的LoRA路径或默认路径
        if lora_path:
            self.lora_path = Path(lora_path)
        else:
            # 根据模式选择不同的默认模型权重路径
            if self.classifier_head:
                self.lora_path = self.config.sft_save_dir / "final_model_classifier"
            else:
                self.lora_path = self.config.sft_save_dir / "final_model_gen"
        self.templates = PipelineClassificationTemplates()
        
        # 设置随机种子以确保可重复性
        self._set_seed(seed)
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.classifier_head:
            # 加载分类头配置和权重
            classifier_path = self.lora_path / "classifier.pt"
            if not classifier_path.exists():
                raise FileNotFoundError(f"找不到分类头文件: {classifier_path}")
            
            classifier_save = torch.load(classifier_path, map_location='cpu', weights_only=True)
            classifier_config = classifier_save["config"]
            classifier_state = classifier_save["state_dict"]
            
            # 加载基础模型
            base_model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # 使用保存的配置创建分类模型
            self.model = QwenForSequenceClassification(
                base_model,
                num_labels=classifier_config["num_labels"]
            )
            
            # 加载分类头权重
            self.model.classifier.load_state_dict(classifier_state)
            
            # 转换为float16
            self.model.classifier = self.model.classifier.half()
        else:
            # 加载生成式模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(
            self.model,
            self.lora_path,
            torch_dtype=torch.float16
        )
        
        # 设置为评估模式
        self.model.eval()
        
        # 禁用dropout等随机行为
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
                
    def _predict_with_head(self, question: str, schema: dict) -> tuple[int, dict]:
        """使用分类头模型进行预测"""
        input_text = self.templates.create_classifier_prompt(question, schema)
        # print(input_text)
        inputs = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.model.device, dtype=torch.long) for k, v in inputs.items()}
        
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 使用softmax获取概率分布
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            
            # 获取每个类别的概率
            probabilities = {
                "basic": float(probs[0][0]),
                "intermediate": float(probs[0][1]),
                "advanced": float(probs[0][2])
            }
        # print("[QwenRouter] predicted_class:", predicted_class, "probabilities:", probabilities)
        return predicted_class, probabilities
        
    def _predict_with_gen(self, question: str, schema: dict) -> int:
        """使用生成式模型进行预测"""
        prompt = self.templates.create_generator_prompt(question, schema)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        try:
            label = int(response)
            if label not in [1, 2, 3]:
                label = 3
            return label - 1  # 转换为0-based
        except:
            return 2  # 默认使用最复杂的pipeline
        
    def _predict(self, question: str, schema: dict) -> int:
        """根据模式选择预测方法"""
        if self.classifier_head:
            return self._predict_with_head(question, schema)
        else:
            return self._predict_with_gen(question, schema)
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """
        根据Qwen模型的预测结果选择合适的SQL生成器
        
        Args:
            query: 用户查询
            schema_linking_output: Schema Linking的输出
            query_id: 查询ID
            
        Returns:
            str: 选择的生成器名称 (对应 PipelineLevel)
        """
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # 使用模型进行预测
        if self.classifier_head:
            predicted_class, probabilities = self._predict_with_head(query, linked_schema)
            
            # 记录预测概率
            self.logger.info(f"Pipeline selection probabilities for query {query_id}:")
            for pipeline, prob in probabilities.items():
                self.logger.info(f"  {pipeline}: {prob:.4f}")
        else:
            predicted_class = self._predict_with_gen(query, linked_schema)
        
        # 根据预测结果选择pipeline
        if predicted_class == 0:  # Basic
            return PipelineLevel.BASIC.value
        elif predicted_class == 1:  # Intermediate
            return PipelineLevel.INTERMEDIATE.value
        elif predicted_class == 2:  # Advanced
            return PipelineLevel.ADVANCED.value
        else:
            raise ValueError(f"Invalid prediction from classifier: {predicted_class}")

