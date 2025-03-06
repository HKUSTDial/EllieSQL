import torch
import torch.nn.functional as F
from typing import Dict
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..sft.qwen_cascade_sft import QwenForBinaryClassification
from ..sft.instruction_templates import PipelineClassificationTemplates

class QwenCascadeClassifier:
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
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.qwen_dir,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载分类头配置和权重
        classifier_path = self.model_path / "classifier.pt"
        if not classifier_path.exists():
            raise FileNotFoundError(f"找不到分类头文件: {classifier_path}")
        
        classifier_save = torch.load(classifier_path, map_location='cpu', weights_only=True)
        classifier_config = classifier_save["config"]
        classifier_state = classifier_save["state_dict"]
        
        # 加载基础模型
        base_model = AutoModel.from_pretrained(
            self.config.qwen_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 使用保存的配置创建分类模型
        self.model = QwenForBinaryClassification(base_model)
        
        # 加载分类头权重
        self.model.classifier.load_state_dict(classifier_state)
        
        # 转换为float16
        self.model.classifier = self.model.classifier.half()
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(
            self.model,
            self.model_path,
            torch_dtype=torch.float16
        )
        
        # 设置为评估模式
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
        
        # 将输入移动到正确的设备和数据类型
        inputs = {k: v.to(self.model.device, dtype=torch.long) for k, v in inputs.items()}
        
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

class QwenCascadeRouter(RouterBase):
    """级联式Qwen路由器：按basic->intermediate->advanced顺序判断"""
    
    def __init__(
        self,
        name: str = "QwenCascadeRouter",
        confidence_threshold: float = 0.5,
        model_path: str = None,
        seed: int = 42
    ):
        super().__init__(name)
        self.config = Config()
        self.model_path = Path(model_path) if model_path else self.config.cascade_qwen_save_dir
        
        # 设置随机种子
        self._set_seed(seed)
        
        # 初始化pipeline对应的分类器
        self.basic_classifier = QwenCascadeClassifier(
            pipeline_type="basic",
            model_name="basic_classifier",
            model_path=self.model_path,
            confidence_threshold=confidence_threshold
        )
        print("[i] Successfully load and initialize classifier for basic pipeline")
        
        self.intermediate_classifier = QwenCascadeClassifier(
            pipeline_type="intermediate",
            model_name="intermediate_classifier",
            model_path=self.model_path,
            confidence_threshold=confidence_threshold
        )
        print("[i] Successfully load and initialize classifier for intermediate pipeline")
        
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
        self.logger.info(f"Query {query_id} routed to ADVANCED pipeline (confidence: {confidence:.3f})")
        return PipelineLevel.ADVANCED.value 