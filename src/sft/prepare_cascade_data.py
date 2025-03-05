import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from ..core.config import Config
from .instruction_templates import PipelineClassificationTemplates
from ..core.utils import load_jsonl
import argparse

class CascadeDataProcessor:
    """为级联式分类器准备二分类数据集"""
    
    def __init__(self, cascade_dataset: str, seed: int = 42):
        self.config = Config()
        self.cascade_dataset_dir = self.config.cascade_data_dir / cascade_dataset
        self.cascade_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.templates = PipelineClassificationTemplates()
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(self.seed)
        
    def prepare_data(self, labeled_file: str, train_ratio: float = 0.8):
        """准备三个pipeline的二分类数据集"""
        # 加载数据
        data = load_jsonl(labeled_file)
        
        # 为每个pipeline准备数据
        basic_samples = self._prepare_basic_samples(data)
        intermediate_samples = self._prepare_intermediate_samples(data)
        advanced_samples = self._prepare_advanced_samples(data)
        
        # 为每个pipeline划分并保存数据
        self._process_pipeline_data("basic", basic_samples, train_ratio)
        self._process_pipeline_data("intermediate", intermediate_samples, train_ratio)
        self._process_pipeline_data("advanced", advanced_samples, train_ratio)
        
    def _prepare_basic_samples(self, data: List[Dict]) -> List[Dict]:
        """准备basic pipeline的二分类样本"""
        samples = []
        for item in data:
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            # basic标签的问题标记为1，其他标记为0
            label = 1 if item["label"] == 1 else 0
            
            samples.append({
                "text": input_text,
                "label": label,
                "question_id": item["question_id"],
                "source": item.get("source", "unknown"),
                "original_label": item["label"]
            })
        return samples
        
    def _prepare_intermediate_samples(self, data: List[Dict]) -> List[Dict]:
        """准备intermediate pipeline的二分类样本"""
        samples = []
        for item in data:
            # 忽略basic标签的问题
            if item["label"] == 1:
                continue
                
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            # intermediate标签的问题标记为1，advanced和unsolved标记为0
            label = 1 if item["label"] == 2 else 0
            
            samples.append({
                "text": input_text,
                "label": label,
                "question_id": item["question_id"],
                "source": item.get("source", "unknown"),
                "original_label": item["label"]
            })
        return samples
        
    def _prepare_advanced_samples(self, data: List[Dict]) -> List[Dict]:
        """准备advanced pipeline的二分类样本"""
        samples = []
        for item in data:
            # 忽略basic和intermediate标签的问题
            if item["label"] in [1, 2]:
                continue
                
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            # advanced标签的问题标记为1，unsolved标记为0
            label = 1 if item["label"] == 3 else 0
            
            samples.append({
                "text": input_text,
                "label": label,
                "question_id": item["question_id"],
                "source": item.get("source", "unknown"),
                "original_label": item["label"]
            })
        return samples
        
    def _process_pipeline_data(self, pipeline: str, samples: List[Dict], train_ratio: float):
        """处理单个pipeline的数据：划分训练集和验证集，保存并分析分布"""
        # 准备分层采样的标签
        labels = [s["label"] for s in samples]
        
        # 使用分层采样划分数据集
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1-train_ratio,
            random_state=self.seed
        )
        
        # 获取划分索引
        indices = np.arange(len(labels))
        for train_idx, valid_idx in sss.split(indices, labels):
            train_samples = [samples[i] for i in train_idx]
            valid_samples = [samples[i] for i in valid_idx]
            
        # 保存数据集
        train_file = self.cascade_dataset_dir / f"{pipeline}_train.json"
        valid_file = self.cascade_dataset_dir / f"{pipeline}_valid.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
            
        # 分析并打印分布
        print(f"\n=== {pipeline.upper()} Pipeline ===")
        print(f"Total samples: {len(samples)}")
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(valid_samples)}")
        
        # 分析标签分布
        def get_label_dist(samples):
            total = len(samples)
            pos_count = sum(1 for s in samples if s["label"] == 1)
            return {
                "positive": pos_count,
                "negative": total - pos_count,
                "pos_ratio": pos_count/total*100
            }
            
        train_dist = get_label_dist(train_samples)
        valid_dist = get_label_dist(valid_samples)
        
        print("\nLabel Distribution:")
        print("Training set:")
        print(f"  Positive: {train_dist['positive']} ({train_dist['pos_ratio']:.1f}%)")
        print(f"  Negative: {train_dist['negative']} ({100-train_dist['pos_ratio']:.1f}%)")
        print("Validation set:")
        print(f"  Positive: {valid_dist['positive']} ({valid_dist['pos_ratio']:.1f}%)")
        print(f"  Negative: {valid_dist['negative']} ({100-valid_dist['pos_ratio']:.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cascade_dataset', type=str, required=True,
                       help='Name of the specified cascade dataset directory under data/cascade/')
    parser.add_argument('--labeled_file', type=str, required=True,
                       help='Path to the labeled data file')
    args = parser.parse_args()
    
    processor = CascadeDataProcessor(cascade_dataset=args.cascade_dataset, seed=42)
    processor.prepare_data(args.labeled_file)

if __name__ == "__main__":
    main() 