import json
from pathlib import Path
from typing import Dict, List
import random
from ..core.config import Config
from .instruction_templates import PipelineClassificationTemplates

class ClassifierDataProcessor:
    """处理标注数据生成微调数据集 (用于添加了分类头模型的分类任务)"""
    
    def __init__(self):
        self.config = Config()
        self.finetune_dir = self.config.data_dir / "finetune"
        self.finetune_dir.mkdir(parents=True, exist_ok=True)
        self.templates = PipelineClassificationTemplates()
        
    def prepare_data(self, labeled_file: str, train_ratio: float = 0.8):
        """准备分类模型的训练和验证数据集"""
        # 加载数据
        with open(labeled_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        # 构造分类样本
        samples = []
        for item in data:
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            label = item["label"] - 1  # 转换为0-based标签
            
            samples.append({
                "text": input_text,
                "label": label,
                "question_id": item["question_id"]
            })
            
        # 处理和保存数据
        self._process_and_save_samples(samples, train_ratio)
        
    def _process_and_save_samples(self, samples: List[Dict], train_ratio: float):
        """处理和保存数据集"""
        random.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        valid_samples = samples[split_idx:]
        
        # 保存数据集
        train_file = self.finetune_dir / "classifier_train.json"
        valid_file = self.finetune_dir / "classifier_valid.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
            
        # 打印统计信息
        print(f"Total samples: {len(samples)}")
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(valid_samples)}")
        print(f"Data saved to {self.finetune_dir}")

def main():
    processor = ClassifierDataProcessor()
    labeled_file = "data/labeled/sampled_bird_demo_pipeline_preference.jsonl"
    processor.prepare_data(labeled_file)

if __name__ == "__main__":
    main() 