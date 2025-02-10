import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from ..core.config import Config
from .instruction_templates import PipelineClassificationTemplates
from ..core.utils import load_jsonl

class GeneratorDataProcessor:
    """处理标注数据生成微调数据集 (用于生成式无分类头的模型)"""
    
    def __init__(self, seed: int = 42):
        self.config = Config()
        self.sft_data_dir = self.config.sft_data_dir
        self.sft_data_dir.mkdir(parents=True, exist_ok=True)
        self.templates = PipelineClassificationTemplates()
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(self.seed)
        
    def prepare_data(self, labeled_file: str, train_ratio: float = 0.8):
        """准备生成式模型的训练和验证数据集"""
        # 加载数据
        data = load_jsonl(labeled_file)
        
        # 构造生成样本
        samples = []
        labels = []  # 用于分层采样
        for item in data:
            prompt = self.templates.create_generator_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            
            # 将标签4转换为3
            label = min(item["label"], 3)
            
            samples.append({
                "prompt": prompt,
                "response": str(label),
                "question_id": item["question_id"],
                "source": item["source"],
                "difficulty": item["difficulty"],
                "label": label  # 保存用于分析，不参与训练
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
        # 保存数据集
        train_file = self.sft_data_dir / "gen_train.json"
        valid_file = self.sft_data_dir / "gen_valid.json"
        
        # 移除分析用的label字段
        train_save = [{k: v for k, v in s.items() if k != 'label'} for s in train_samples]
        valid_save = [{k: v for k, v in s.items() if k != 'label'} for s in valid_samples]
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_save, f, ensure_ascii=False, indent=2)
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_save, f, ensure_ascii=False, indent=2)
            
        # 分析分布
        def analyze_distribution(samples: List[Dict]) -> Dict:
            total = len(samples)
            # 标签分布
            label_dist = {}
            for i in range(1, 4):  # 1, 2, 3
                count = sum(1 for s in samples if s["label"] == i)
                label_dist[f"Label {i}"] = {
                    "count": count,
                    "percentage": count/total*100
                }
                
            # 难度分布
            difficulty_dist = {}
            for s in samples:
                diff = s["difficulty"]
                if diff not in difficulty_dist:
                    difficulty_dist[diff] = 0
                difficulty_dist[diff] += 1
                
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
                "difficulties": difficulty_dist,
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
            
        print("\nDifficulty Distribution:")
        print("Training set:", train_dist["difficulties"])
        print("Validation set:", valid_dist["difficulties"])
        
        print("\nSource Distribution:")
        print("Training set:", train_dist["sources"])
        print("Validation set:", valid_dist["sources"])
        
        print(f"\nData saved to {self.sft_data_dir}")

def main():
    processor = GeneratorDataProcessor(seed=42)
    labeled_file = "data/labeled/bird_dev_pipeline_label.jsonl"
    processor.prepare_data(labeled_file)

if __name__ == "__main__":
    main() 