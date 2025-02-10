from typing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..finetune.qwen_classifier_sft import QwenForSequenceClassification
from ..finetune.instruction_templates import PipelineClassificationTemplates

class QwenClassifierRouter(RouterBase):
    """添加了分类头并使用LoRA SFT的Qwen路由器"""
    
    def __init__(self, name: str = "QwenClassifierRouter", seed: int = 42):
        super().__init__(name)
        self.config = Config()
        self.model_path = self.config.model_dir
        self.lora_path = self.config.finetune_save_dir / "final_model_classifier"
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
                
    def _predict(self, question: str, schema: dict) -> tuple[int, dict]:
        """使用分类头模型进行预测"""
        input_text = self.templates.create_classifier_prompt(question, schema)
        
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
            
        return predicted_class, probabilities
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """根据分类器预测结果选择合适的SQL生成器"""
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # 使用分类器进行预测
        predicted_class, probabilities = self._predict(query, linked_schema)
        
        # 记录预测概率
        self.logger.info(f"Pipeline selection probabilities for query {query_id}:")
        for pipeline, prob in probabilities.items():
            self.logger.info(f"  {pipeline}: {prob:.4f}")
        
        # 根据预测结果选择pipeline
        if predicted_class == 0:  # Basic
            return PipelineLevel.BASIC.value
        elif predicted_class == 1:  # Intermediate
            return PipelineLevel.INTERMEDIATE.value
        elif predicted_class == 2:  # Advanced
            return PipelineLevel.ADVANCED.value
        else:
            raise ValueError(f"Invalid prediction from classifier: {predicted_class}") 