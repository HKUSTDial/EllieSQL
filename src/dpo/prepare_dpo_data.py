import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from ..core.config import Config
from ..sft.instruction_templates import PipelineClassificationTemplates
from ..core.utils import load_jsonl
import argparse

class DPODataProcessor:
    """处理标注数据生成DPO训练数据集
    
    生成的数据集格式:
    {
        "question_id": str,          # 问题ID
        "input": str,                # 输入文本(问题+schema)
        "chosen": str,               # 偏好的pipeline类型 ("basic"/"intermediate"/"advanced")
        "rejected": str,             # 不偏好的pipeline类型
        "source": str,               # 数据来源
        "original_label": int        # 原始标签(1-4)
    }
    
    偏序关系说明:
    - Basic (1): 无偏好关系
    - Intermediate (2): 偏好intermediate > basic > advanced
    - Advanced (3/4): 偏好advanced > intermediate/basic
    """
    
    def __init__(self, dpo_dataset: str, seed: int = 42):
        self.config = Config()
        self.dpo_dataset_dir = self.config.pairwise_data_dir / dpo_dataset
        self.dpo_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.templates = PipelineClassificationTemplates()
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(self.seed)
        
        # 定义pipeline类型
        self.pipeline_types = ["basic", "intermediate", "advanced"]
        self.label_pairs = None  # 初始化为None，需要通过set_label_pairs设置
        
    def _validate_preference_pairs(self, pairs: Dict[int, List[Tuple[str, str]]]):
        """验证偏序对的合法性"""
        valid_responses = {"basic", "intermediate", "advanced"}
        for label, preference_list in pairs.items():
            for chosen, rejected in preference_list:
                if chosen not in valid_responses or rejected not in valid_responses:
                    raise ValueError(f"Invalid response in pair ({chosen}, {rejected}) for label {label}")
                if chosen == rejected:
                    raise ValueError(f"Self-preference found in pair ({chosen}, {rejected}) for label {label}")

    def set_label_pairs(self, label_pairs: Dict[int, List[Tuple[str, str]]]):
        """设置每个标签对应的偏序对"""
        self._validate_preference_pairs(label_pairs)
        self.label_pairs = label_pairs
        
    def _create_preference_pairs(self, label: int) -> List[Tuple[str, str]]:
        """根据标签创建偏序对"""
        if not self.label_pairs:
            raise ValueError("Label pairs not set. Call set_label_pairs first.")
        if label not in self.label_pairs:
            raise ValueError(f"Invalid label: {label}")
        return self.label_pairs[label]
        
    def _validate_sample(self, sample: Dict) -> bool:
        """验证样本格式"""
        required_keys = {"question_id", "input", "chosen", "rejected", 
                        "source", "original_label"}
        
        # 检查必需字段
        if not all(key in sample for key in required_keys):
            return False
        
        # 验证标签范围
        if not 1 <= sample["original_label"] <= 4:
            return False
        
        # 验证chosen和rejected是否为有效的pipeline类型
        valid_types = {"basic", "intermediate", "advanced"}
        if sample["chosen"] not in valid_types or sample["rejected"] not in valid_types:
            return False
        
        return True

    def prepare_data(self, labeled_file: str, train_ratio: float = 0.8):
        """准备DPO训练数据集"""
        # 加载数据
        data = load_jsonl(labeled_file)
        
        # 构造DPO样本
        dpo_samples = []
        for item in data:
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            
            # 获取标签(1-based)
            label = item["label"]
            
            # 生成偏序对
            preference_pairs = self._create_preference_pairs(label)
            
            # 为每个偏序对创建一个样本
            for chosen, rejected in preference_pairs:
                dpo_samples.append({
                    "question_id": item["question_id"],
                    "input": input_text,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": item.get("source", "unknown"),
                    "original_label": label
                })
        
        # 随机打乱数据
        np.random.shuffle(dpo_samples)
        
        # 划分训练集和验证集
        split_idx = int(len(dpo_samples) * train_ratio)
        train_samples = dpo_samples[:split_idx]
        valid_samples = dpo_samples[split_idx:]
        
        # 验证所有样本
        invalid_samples = [i for i, sample in enumerate(dpo_samples) 
                          if not self._validate_sample(sample)]
        if invalid_samples:
            raise ValueError(f"Invalid samples found at indices: {invalid_samples}")
        
        # 保存数据集
        self._save_and_analyze_samples(train_samples, valid_samples)
        
    def _analyze_sample_lengths(self, samples: List[Dict]) -> Dict:
        """分析样本长度分布"""
        input_lengths = [len(s["input"]) for s in samples]
        return {
            "min_length": min(input_lengths),
            "max_length": max(input_lengths),
            "avg_length": sum(input_lengths) / len(input_lengths)
        }

    def _save_and_analyze_samples(self, train_samples: List[Dict], valid_samples: List[Dict]):
        """保存数据集并分析分布"""
        # 保存数据集
        train_file = self.dpo_dataset_dir / "dpo_train.json"
        valid_file = self.dpo_dataset_dir / "dpo_valid.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
            
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
        valid_dist = analyze_distribution(valid_samples)
        
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
            
        # 添加长度分析
        train_lengths = self._analyze_sample_lengths(train_samples)
        valid_lengths = self._analyze_sample_lengths(valid_samples)
        
        print("\nSample Length Analysis:")
        print("Training set:")
        print(f"  Min length: {train_lengths['min_length']}")
        print(f"  Max length: {train_lengths['max_length']}")
        print(f"  Avg length: {train_lengths['avg_length']:.1f}")
        print("Validation set:")
        print(f"  Min length: {valid_lengths['min_length']}")
        print(f"  Max length: {valid_lengths['max_length']}")
        print(f"  Avg length: {valid_lengths['avg_length']:.1f}")
        
        print(f"\nData saved to {self.dpo_dataset_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo_dataset', type=str, required=True,
                       help='Name of the specified DPO dataset directory under data/dpo/')
    parser.add_argument('--labeled_file', type=str, required=True,
                       help='Path to the labeled data file')
    args = parser.parse_args()
    
    processor = DPODataProcessor(dpo_dataset=args.dpo_dataset, seed=42)
    
    # 定义每个标签的偏序对
    label_pairs = {
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
            ("advanced", "basic"),
            ("advanced", "intermediate")
        ],
        4: [  # Unsolved (same as Advanced)
            ("advanced", "basic"),
            ("advanced", "intermediate")
        ]
    }
    
    processor.set_label_pairs(label_pairs)
    processor.prepare_data(args.labeled_file)

if __name__ == "__main__":
    main()