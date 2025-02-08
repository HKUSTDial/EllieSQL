import json
import random
from pathlib import Path
from typing import List, Dict
import argparse
from ..core.utils import load_jsonl
from .labeler import PipelineType

class LabeledDataSampler:
    """用于从标记数据集中采样指定标签的样本"""
    
    @staticmethod
    def load_labeled_data(file_path: str) -> List[Dict]:
        """加载标记数据"""
        return load_jsonl(file_path)
    
    @staticmethod
    def sample_by_label(data: List[Dict], label: int, sample_size: int) -> List[Dict]:
        """从指定标签的数据中随机采样"""
        # 筛选指定标签的数据
        labeled_data = [item for item in data if item["label"] == label]
        
        # 如果数据量小于采样数，返回全部数据
        if len(labeled_data) <= sample_size:
            return labeled_data
            
        # 随机采样
        return random.sample(labeled_data, sample_size)
    
    @staticmethod
    def print_samples(samples: List[Dict]):
        """打印采样结果"""
        for i, sample in enumerate(samples, 1):
            print("\n" + "="*80)
            print(f"[Sample {i}/{len(samples)}]")
            # print("="*80)
            print(f"Question ID: {sample['question_id']}")
            # print(f"Source: {sample['source']}")
            print(f"Difficulty: {sample['difficulty']}; Pipeline Label: {sample['label']}")
            # print(f"Question: {sample['question']}")
            # print(f"Pipeline Type: {sample['pipeline_type']}")
            # print(f"DB ID: {sample['db_id']}")
            # print("\nSchema:")
            # for table in sample['enhanced_linked_schema_wo_info']['tables']:
            #     print(f"  Table: {table['table']}")
            #     print(f"  Columns: {', '.join(table['columns'])}")
            #     if 'primary_keys' in table:
            #         print(f"  Primary Keys: {', '.join(table['primary_keys'])}")
            #     print()
            print(f"Gold SQL: {sample['gold_sql']}")
            print("="*80)

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Sample labeled data by pipeline type')
    parser.add_argument('file_path', type=str, help='Path to the labeled data file')
    parser.add_argument('--label', type=int, choices=[1, 2, 3, 4], 
                      help='Label to sample (1: BASIC, 2: INTERMEDIATE, 3: ADVANCED, 4: UNSOLVED)')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of samples to show')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 加载数据
    sampler = LabeledDataSampler()
    data = sampler.load_labeled_data(args.file_path)
    
    # 获取标签名称
    label_name = PipelineType(args.label).name
    
    # 统计该标签的总数
    total_count = sum(1 for item in data if item["label"] == args.label)
    
    print(f"\nSampling {args.sample_size} examples from {label_name} (Total: {total_count})")
    
    # 采样并打印
    samples = sampler.sample_by_label(data, args.label, args.sample_size)
    sampler.print_samples(samples)
    
    print(f"\nShowed {len(samples)} samples out of {total_count} {label_name} examples")

if __name__ == "__main__":
    main() 