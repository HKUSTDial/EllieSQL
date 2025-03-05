import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
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

class RoBERTaCascadeClassifier:
    """用于二分类推理的RoBERTa分类器"""

    def __init__(self, pipeline_type: str, model_name: str, confidence_threshold: float = 0.5, seed: int = 42):
        # 设置随机种子
        set_seed(seed)
        
        self.config = Config()
        self.pipeline_type = pipeline_type
        self.model_path = self.config.cascade_roberta_save_dir / f"final_model_{model_name}"
        self.templates = PipelineClassificationTemplates()
        
        # 添加置信度阈值参数
        self.confidence_threshold = confidence_threshold
        
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

def test(pipeline_type: str, confidence_threshold: float):
    """测试二分类器推理"""

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

    # 选择要测试的pipeline类型
    model_name = f"{pipeline_type}_classifier"
    
    classifier = RoBERTaCascadeClassifier(
        pipeline_type=pipeline_type,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        seed=42
    )

    # 对每个测试用例进行分类
    print(f"\nTesting {pipeline_type.upper()} Pipeline Classifier (confidence threshold: {confidence_threshold}):")
    print(f"Expected: True for basic questions, False for others" if pipeline_type == "basic" else
          f"Expected: True for intermediate questions, False for advanced/unsolved" if pipeline_type == "intermediate" else
          f"Expected: True for advanced questions, False for unsolved")
    
    for i, test_case in enumerate(test_cases, 1):
        can_handle, confidence = classifier.classify(test_case["question"], test_case["schema"])
        print(f"[Test Case {i}] Can handle by {pipeline_type} pipeline: {can_handle} "
              f"(confidence: {confidence:.3f}, threshold: {confidence_threshold})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_type", type=str, default="basic", 
                       choices=["basic", "intermediate", "advanced"])
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Confidence threshold for positive predictions")
    args = parser.parse_args()
    
    test(args.pipeline_type, args.confidence_threshold) 