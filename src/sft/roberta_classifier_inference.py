import json
from pathlib import Path
import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from ..core.config import Config
from .instruction_templates import PipelineClassificationTemplates

def set_seed(seed: int = 42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 禁用cudnn的随机性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RoBERTaClassifier:
    """用于分类推理的RoBERTa分类器"""

    def __init__(self, model_path: str = None, seed: int = 42):
        # 设置随机种子
        set_seed(seed)
        
        self.config = Config()
        self.model_path = Path(model_path) if model_path else self.config.roberta_save_dir / "final_model_roberta_lora"
        self.templates = PipelineClassificationTemplates()
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
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
        
    def classify(self, question: str, schema: dict) -> tuple[int, dict]:
        """进行分类预测，返回预测的类别和概率分布"""
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

def test():
    """测试分类器推理"""
    classifier = RoBERTaClassifier(
        model_path="/data/zhuyizhang/saves/RoBERTa-router/final_model_roberta_lora",
        seed=42
    )
    
    # 测试样例
    test_cases = [
        {
            "question": "How many pets have a greater weight than 10?",
            "schema": {
                "tables": [{
                    "table": "Pets",
                    "columns": ["PetID", "weight"],
                    "primary_keys": ["PetID"]
                }]
            }
        },
        {
            "question": "Give the publisher ID of Star Trek.",
            "schema": {
                "tables": [{
                    "table": "publisher",
                    "columns": ["publisher_name", "id"],
                    "primary_keys": ["id"]
                }]
            }
        },
        {
            "question": "What is the border color of card \"Ancestor's Chosen\"?",
            "schema": {
                "tables": [{
                    "table": "cards",
                    "columns": ["id", "borderColor", "name"],
                    "primary_keys": ["id"]
                }]
            }
        },
        {
            "question": "How many patients with an Ig G higher than normal?",
            "schema": {
                "tables": [{
                    "table": "Laboratory",
                    "columns": ["ID", "IGG", "Date"],
                    "primary_keys": ["ID", "Date"]
                }, {
                    "table": "Patient",
                    "columns": ["ID"],
                    "primary_keys": ["ID"]
                }]
            }
        },
        {
            "question": "What is the eligible free or reduced price meal rate for the top 5 schools in grades 1-12 with the highest free or reduced price meal count of the schools with the ownership code 66?",
            "schema": {
                "tables": [{
                    "table": "frpm",
                    "columns": ["Enrollment (K-12)", "CDSCode", "FRPM Count (K-12)"],
                    "primary_keys": ["CDSCode"]
                }, {
                    "table": "schools",
                    "columns": ["SOC", "CDSCode"],
                    "primary_keys": ["CDSCode"]
                }]
            }
        }
    ]

    # 对每个测试用例进行分类
    print(f"Supposed to be: 1, 1, 1, 3, 3")
    for i, test_case in enumerate(test_cases, 1):
        label, probabilities = classifier.classify(test_case["question"], test_case["schema"])
        print(f"[Test Case {i}] Predicted Label: {label}, Probabilities: {probabilities}")

if __name__ == "__main__":
    test() 