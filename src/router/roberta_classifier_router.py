from typing import Dict
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..sft.instruction_templates import PipelineClassificationTemplates
from pathlib import Path

class RoBERTaClassifierRouter(RouterBase):
    """使用RoBERTa进行分类的路由器"""
    
    def __init__(self, name: str = "RoBERTaClassifierRouter", model_path: str = None, seed: int = 42):
        super().__init__(name)
        self.config = Config()
        # 使用指定的模型路径或默认路径
        self.model_path = Path(model_path) if model_path else self.config.roberta_save_dir / "final_model_roberta"
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
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}")
            
        # 从原始预训练模型加载tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.config.roberta_dir,  # 使用预训练模型的tokenizer
            padding_side="right"
        )
        
        # 加载微调后的分类模型
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=3,  # Basic, Intermediate, Advanced
            problem_type="single_label_classification"
        )
        
        # 将模型移动到GPU并设置为评估模式
        self.model = self.model.cuda()
        self.model.eval()
        
        # 禁用dropout等随机行为
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
                
    def _predict(self, question: str, schema: dict) -> tuple[int, dict]:
        """使用RoBERTa模型进行预测"""
        # 使用模板创建输入文本
        input_text = self.templates.create_classifier_prompt(question, schema)
        
        # 编码输入
        inputs = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 将输入移动到GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 进行预测
        with torch.no_grad():
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
            
        # 转换回1-based标签
        return predicted_class + 1, probabilities
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """根据RoBERTa分类器预测结果选择合适的SQL生成器"""
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # 使用分类器进行预测
        predicted_class, probabilities = self._predict(query, linked_schema)
        
        # 记录预测概率
        self.logger.info(f"Pipeline selection probabilities for query {query_id}:")
        for pipeline, prob in probabilities.items():
            self.logger.info(f"  {pipeline}: {prob:.4f}")
        
        # 根据预测结果选择pipeline
        if predicted_class == 1:  # Basic
            return PipelineLevel.BASIC.value
        elif predicted_class == 2:  # Intermediate
            return PipelineLevel.INTERMEDIATE.value
        elif predicted_class == 3:  # Advanced
            return PipelineLevel.ADVANCED.value
        else:
            raise ValueError(f"Invalid prediction from classifier: {predicted_class}") 