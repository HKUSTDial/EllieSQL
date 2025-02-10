import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from ..core.config import Config
from .qwen_classifier_sft import QwenForSequenceClassification
from .instruction_templates import PipelineClassificationTemplates
import json

def set_seed(seed: int = 42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 禁用cudnn的随机性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class QwenClassifier:
    """用于分类推理的Qwen分类器: """

    def __init__(self, seed: int = 42):
        # 设置随机种子
        set_seed(seed)
        
        self.config = Config()
        self.model_path = self.config.model_dir
        self.lora_path = self.config.sft_save_dir / "final_model_classifier"
        self.templates = PipelineClassificationTemplates()
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
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
        
    def classify(self, question: str, schema: dict) -> tuple[int, dict]:
        """进行分类预测，返回预测的类别和概率分布"""
        # 使用模板创建输入文本
        input_text = self.templates.create_classifier_prompt(question, schema)
        # print(input_text)
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
            
        # 转换回1-based标签
        return predicted_class + 1, probabilities

def test_deterministic():
    """测试分类结果的确定性"""
    classifier = QwenClassifier(seed=42)
    
    # 测试样例
    question = "Among the weekly issuance accounts, how many have a loan of under 200000?"
    # Table: account; Columns: frequency, account_id\nTable: loan; Columns: amount, account_id, loan_id\
    schema = {
        "tables": [{
            "table": "account",
            "columns": ["frequency", "account_id"],
            "primary_keys": ["account_id"]
        }, {
            "table": "loan",
            "columns": ["amount", "account_id", "loan_id"],
            "primary_keys": ["account_id"]
        }]
    }
    
    # 多次运行分类，确保结果一致
    label, probabilities = classifier.classify(question, schema)
    print(f"Predicted Label: {label}, Probabilities: {probabilities}")
    

if __name__ == "__main__":
    test_deterministic() 