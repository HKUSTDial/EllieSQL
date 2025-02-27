import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from ..core.config import Config
from .instruction_templates import PipelineClassificationTemplates
from ..core.utils import load_jsonl
import argparse

class ClassifierDataProcessor:
    """处理标注数据生成微调数据集 (用于添加了分类头模型的分类任务)"""
    
    def __init__(self, sft_dataset: str, seed: int = 42):
        self.config = Config()
        # 在sft_data_dir下创建指定的数据集目录
        self.sft_dataset_dir = self.config.sft_data_dir / sft_dataset
        self.sft_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.templates = PipelineClassificationTemplates()
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(self.seed)
        
    def prepare_data(self, labeled_file: str, train_ratio: float = 0.8):
        """准备分类模型的训练和验证数据集"""
        # 加载数据
        data = load_jsonl(labeled_file)
        
        # 构造分类样本
        samples = []
        labels = []  # 用于分层采样
        for item in data:
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            # 将标签4转换为3，然后转为0-based
            label = min(item["label"], 3) - 1
            
            samples.append({
                "text": input_text,
                "label": label,
                "question_id": item["question_id"],
                "source": item.get("source", "unknown")
            })
            labels.append(label)
            
        # 使用分层采样划分数据集
        train_indices, valid_indices = self._stratified_split(
            labels, train_ratio=train_ratio
        )
        
        train_samples = [samples[i] for i in train_indices]
        valid_samples = [samples[i] for i in valid_indices]
        
        # 保存数据集
        self._save_and_analyze_samples(train_samples, valid_samples)
        
    def _stratified_split(self, labels: List[int], train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """使用分层采样划分数据集"""
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1-train_ratio,
            random_state=self.seed
        )
        
        # 获取划分索引
        indices = np.arange(len(labels))
        for train_idx, valid_idx in sss.split(indices, labels):
            return train_idx, valid_idx
            
    def _save_and_analyze_samples(self, train_samples: List[Dict], valid_samples: List[Dict]):
        """保存数据集并分析分布"""
        # 保存数据集到指定的数据集目录
        train_file = self.sft_dataset_dir / "classifier_train.json"
        valid_file = self.sft_dataset_dir / "classifier_valid.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
            
        # 分析分布
        def analyze_distribution(samples: List[Dict]) -> Dict:
            total = len(samples)
            # 标签分布
            label_dist = {}
            for i in range(3):  # 0, 1, 2
                count = sum(1 for s in samples if s["label"] == i)
                label_dist[f"Label {i}"] = {
                    "count": count,
                    "percentage": count/total*100
                }
                
            # 来源分布
            source_dist = {}
            for s in samples:
                source = s["source"]
                if source not in source_dist:
                    source_dist[source] = 0
                source_dist[source] += 1
                
            return {
                "total": total,
                "labels": label_dist,
                "sources": source_dist
            }
            
        train_dist = analyze_distribution(train_samples)
        valid_dist = analyze_distribution(valid_samples)
        
        # 打印分析结果
        print(f"\nTotal samples: {len(train_samples) + len(valid_samples)}")
        print(f"Training samples: {train_dist['total']}")
        print(f"Validation samples: {valid_dist['total']}")
        
        print("\nLabel Distribution:")
        print("Training set:")
        for label, stats in train_dist["labels"].items():
            print(f"  {label}: {stats['count']} ({stats['percentage']:.1f}%)")
        print("Validation set:")
        for label, stats in valid_dist["labels"].items():
            print(f"  {label}: {stats['count']} ({stats['percentage']:.1f}%)")
            
        print("\nSource Distribution:")
        print("Training set:", train_dist["sources"])
        print("Validation set:", valid_dist["sources"])
        
        print(f"\nData saved to {self.sft_dataset_dir}")

def main():
    sft_dataset = "bird_train_dataset_simplified"
    labeled_file = "data/labeled/nonempty_bird_train_pipeline_label.jsonl"
    
    processor = ClassifierDataProcessor(sft_dataset=sft_dataset, seed=42)
    processor.prepare_data(labeled_file)

if __name__ == "__main__":
    main() 