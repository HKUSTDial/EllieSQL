import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from ..core.config import Config
from .instruction_templates import PipelineClassificationTemplates
from ..core.utils import load_jsonl
import argparse

class PairwiseDataProcessor:
    """处理标注数据生成pairwise训练数据集"""
    
    def __init__(self, pairwise_dataset: str, seed: int = 42):
        self.config = Config()
        self.pairwise_dataset_dir = self.config.pairwise_data_dir / pairwise_dataset
        self.pairwise_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.templates = PipelineClassificationTemplates()
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(self.seed)
        
        # 定义pipeline类型和对应的索引
        self.pipeline_types = ["basic", "intermediate", "advanced"]
        self.pipeline_to_idx = {
            "basic": 0,
            "intermediate": 1,
            "advanced": 2
        }
        
        # 根据label定义偏序关系
        self.preference_pairs = {
            1: [  # Basic
                ("basic", "intermediate"),
                ("basic", "advanced")
            ],
            2: [  # Intermediate
                ("intermediate", "basic"),
                ("intermediate", "advanced"),
                ("advanced", "basic")
            ],
            3: [  # Advanced
                ("advanced", "intermediate"),
                ("advanced", "basic")
            ],
            4: [  # Unsolved (same as Advanced)
                ("advanced", "intermediate"),
                ("advanced", "basic")
            ]
        }
        
    def prepare_data(self, labeled_file: str, train_ratio: float = 0.8):
        """准备pairwise训练数据集，但保持验证集为原始分类格式"""
        # 加载数据
        data = load_jsonl(labeled_file)
        
        # 随机打乱数据
        np.random.shuffle(data)
        
        # 划分训练集和验证集
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        valid_data = data[split_idx:]
        
        # 构造训练集的pairwise样本
        train_samples = []
        for item in train_data:
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            
            # 获取标签(1-based)并转换为0-based
            label = item["label"] - 1
            
            # 获取该标签对应的偏序对
            preference_pairs = self.preference_pairs[item["label"]]
            
            # 为每个偏序对创建一个样本
            for chosen, rejected in preference_pairs:
                train_samples.append({
                    "question_id": item["question_id"],
                    "input": input_text,
                    "chosen_idx": self.pipeline_to_idx[chosen],
                    "rejected_idx": self.pipeline_to_idx[rejected],
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": item.get("source", "unknown"),
                    "original_label": item["label"]
                })
        
        # 构造验证集（保持原始分类格式以便计算原始准确率）
        valid_samples = []
        for item in valid_data:
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            valid_samples.append({
                "question_id": item["question_id"],
                "input": input_text,
                "label": item["label"] - 1,  # 转换为0-based
                "source": item.get("source", "unknown")
            })
        
        # 保存数据集
        self._save_and_analyze_samples(train_samples, valid_samples)
        
    def _save_and_analyze_samples(self, train_samples: List[Dict], valid_samples: List[Dict]):
        """保存数据集并分析分布"""
        # 保存数据集
        train_file = self.pairwise_dataset_dir / "pairwise_train.json"
        valid_file = self.pairwise_dataset_dir / "pairwise_valid.json"
        
        # 将验证集样本转换为与训练集相同的格式, 生成"伪"的pairwise样本 (用9999表示伪选择)
        formatted_valid_samples = []
        for item in valid_samples:
            # 根据label
            label = item["label"] + 1  # 转回1-based
            # 获取该标签对应的第一个偏序对
            chosen, rejected = self.preference_pairs[label][0]
            
            formatted_valid_samples.append({
                "question_id": item["question_id"],
                "input": item["input"],
                "chosen_idx": 999, #self.pipeline_to_idx[chosen],
                "rejected_idx": 999, #self.pipeline_to_idx[rejected],
                "chosen": chosen,
                "rejected": rejected,
                "source": item["source"],
                "original_label": label,
                "is_validation": True  # 添加标记以区分验证集
            })
        
        # 为训练集添加标记
        for item in train_samples:
            item["is_validation"] = False
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_valid_samples, f, ensure_ascii=False, indent=2)
            
        # 分析分布
        def analyze_distribution(samples: List[Dict]) -> Dict:
            total = len(samples)
            # 原始标签分布
            label_dist = {}
            for i in range(1, 5):  # 1-4
                count = sum(1 for s in samples if s["original_label"] == i)
                label_dist[f"Label {i}"] = {
                    "count": count,
                    "percentage": count/total*100
                }
                
            # 偏序对分布
            pair_dist = {}
            for s in samples:
                pair = (s["chosen"], s["rejected"])
                if pair not in pair_dist:
                    pair_dist[pair] = 0
                pair_dist[pair] += 1
                
            return {
                "total": total,
                "labels": label_dist,
                "pairs": pair_dist
            }
            
        train_dist = analyze_distribution(train_samples)
        valid_dist = analyze_distribution(formatted_valid_samples)
        
        # 打印分析结果
        print(f"\nTotal preference pairs: {len(train_samples) + len(valid_samples)}")
        print(f"Training pairs: {train_dist['total']}")
        print(f"Validation pairs: {valid_dist['total']}")
        
        print("\nOriginal Label Distribution:")
        print("Training set:")
        for label, stats in train_dist["labels"].items():
            print(f"  {label}: {stats['count']} ({stats['percentage']:.1f}%)")
        print("Validation set:")
        for label, stats in valid_dist["labels"].items():
            print(f"  {label}: {stats['count']} ({stats['percentage']:.1f}%)")
            
        print("\nPreference Pair Distribution:")
        print("Training set:")
        for pair, count in train_dist["pairs"].items():
            print(f"  {pair[0]} > {pair[1]}: {count}")
        print("Validation set:")
        for pair, count in valid_dist["pairs"].items():
            print(f"  {pair[0]} > {pair[1]}: {count}")
            
        print(f"\nData saved to {self.pairwise_dataset_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairwise_dataset', type=str, required=True,
                       help='Name of the specified pairwise dataset directory under data/pairwise/')
    parser.add_argument('--labeled_file', type=str, required=True,
                       help='Path to the labeled data file')
    args = parser.parse_args()
    
    processor = PairwiseDataProcessor(pairwise_dataset=args.pairwise_dataset, seed=42)
    processor.prepare_data(args.labeled_file)

if __name__ == "__main__":
    main()
