import torch
import torch.nn.functional as F
from typing import Dict
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..sft.instruction_templates import PipelineClassificationTemplates

class RoBERTaCascadeClassifier:
    """级联式二分类器"""
    
    def __init__(self, pipeline_type: str, model_name: str, model_path: Path, confidence_threshold: float = 0.5):
        self.config = Config()
        self.pipeline_type = pipeline_type
        self.model_path = model_path / f"final_model_{model_name}"
        self.confidence_threshold = confidence_threshold
        self.templates = PipelineClassificationTemplates()
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}")
            
        # 从原始预训练模型加载tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.config.roberta_dir,
            padding_side="right"
        )
        
        # 加载基础模型
        base_model = RobertaForSequenceClassification.from_pretrained(
            self.config.roberta_dir,
            num_labels=2,  # 二分类
            problem_type="single_label_classification"
        )
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(
            base_model,
            self.model_path,
            torch_dtype=torch.float16
        )
        
        # 将模型移动到GPU并设置为评估模式
        self.model = self.model.cuda()
        self.model.eval()
        
        # 禁用dropout等随机行为
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
                
    def classify(self, question: str, schema: dict) -> tuple[bool, float]:
        """进行二分类预测，返回是否可处理和置信度"""
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
            # 获取正类（能处理）的概率
            positive_prob = float(probs[0][1])
            
            # 只有当正类概率超过阈值时才认为能处理
            can_handle = (positive_prob >= self.confidence_threshold)
            
        return can_handle, positive_prob

class RoBERTaCascadeRouter(RouterBase):
    """级联式RoBERTa路由器：按basic->intermediate->advanced顺序判断"""
    
    def __init__(
        self,
        name: str = "RoBERTaCascadeRouter",
        confidence_threshold: float = 0.5,
        model_path: str = None,
        seed: int = 42
    ):
        super().__init__(name)
        self.config = Config()
        self.model_path = Path(model_path) if model_path else self.config.cascade_roberta_save_dir
        
        # 设置随机种子
        self._set_seed(seed)
        
        # 初始化pipeline对应的分类器
        self.basic_classifier = RoBERTaCascadeClassifier(
            pipeline_type="basic",
            model_name="basic_classifier",
            model_path=self.model_path,
            confidence_threshold=confidence_threshold
        )
        print("[i] Successfully load and initialize classifier for basic pipeline")
        
        self.intermediate_classifier = RoBERTaCascadeClassifier(
            pipeline_type="intermediate",
            model_name="intermediate_classifier",
            model_path=self.model_path,
            confidence_threshold=confidence_threshold
        )
        print("[i] Successfully load and initialize classifier for intermediate pipeline")

        # intermediate无法处理的直接使用advanced, 不需要额外判断
        # self.advanced_classifier = RoBERTaCascadeClassifier(
        #     pipeline_type="advanced",
        #     model_name="advanced_classifier",
        #     model_path=self.model_path,
        #     confidence_threshold=confidence_threshold
        # )
        # print("Successfully load and initialize classifier for advanced pipeline")
        
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
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """级联式路由逻辑"""
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # 1. 首先尝试basic pipeline
        can_handle, confidence = self.basic_classifier.classify(query, linked_schema)
        if can_handle:
            self.logger.info(f"Query {query_id} routed to BASIC pipeline (confidence: {confidence:.3f})")
            return PipelineLevel.BASIC.value
            
        # 2. 如果basic不行，尝试intermediate pipeline
        can_handle, confidence = self.intermediate_classifier.classify(query, linked_schema)
        if can_handle:
            self.logger.info(f"Query {query_id} routed to INTERMEDIATE pipeline (confidence: {confidence:.3f})")
            return PipelineLevel.INTERMEDIATE.value
            
        # 3. 如果intermediate不行，直接使用advanced pipeline
        # 可选：也可以先检查advanced pipeline是否能处理
        # can_handle, confidence = self.advanced_classifier.classify(query, linked_schema)
        self.logger.info(
            f"Query {query_id} routed to ADVANCED pipeline (confidence: {confidence:.3f})"
            # f"({'can handle' if can_handle else 'cannot handle'}, confidence: {confidence:.3f})"
        )
        return PipelineLevel.ADVANCED.value
        