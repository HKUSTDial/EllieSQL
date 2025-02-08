import json
import random
from pathlib import Path
from typing import List, Dict
import argparse
from collections import defaultdict
from ..core.utils import load_jsonl
from .labeler import PipelineType

class LabeledDataChecker:
    """用于检查标记数据集的分布并采样查看样本"""
    
    @staticmethod
    def load_labeled_data(file_path: str) -> List[Dict]:
        """加载标记数据"""
        return load_jsonl(file_path)
    
    @staticmethod
    def analyze_distribution(data: List[Dict]) -> Dict:
        """分析数据分布"""
        total = len(data)
        stats = {
            "total": total,
            "labels": defaultdict(int),
            "sources": defaultdict(int),
            "difficulties": defaultdict(int),
            "source_label": defaultdict(lambda: defaultdict(int)),
            "difficulty_label": defaultdict(lambda: defaultdict(int))
        }
        
        for item in data:
            label = item["label"]
            source = item["source"]
            difficulty = item["difficulty"]
            
            # 基本统计
            stats["labels"][label] += 1
            stats["sources"][source] += 1
            stats["difficulties"][difficulty] += 1
            
            # 交叉统计
            stats["source_label"][source][label] += 1
            stats["difficulty_label"][difficulty][label] += 1
            
        return stats
    
    @staticmethod
    def print_distribution(stats: Dict):
        """打印分布统计信息"""
        print("\n" + "="*80)
        print("数据分布统计")
        print("="*80)
        
        # 总体统计
        print(f"\n总样本数: {stats['total']}")
        
        # 标签分布
        print("\n标签分布:")
        for label in sorted(stats["labels"].keys()):
            count = stats["labels"][label]
            percentage = count / stats["total"] * 100
            label_name = PipelineType(label).name
            print(f"  {label_name}: {count} ({percentage:.2f}%)")
            
        # 来源分布
        print("\n来源分布:")
        for source in sorted(stats["sources"].keys()):
            count = stats["sources"][source]
            percentage = count / stats["total"] * 100
            print(f"  {source}: {count} ({percentage:.2f}%)")
            
        # 难度分布
        print("\n难度分布:")
        for difficulty in sorted(stats["difficulties"].keys()):
            count = stats["difficulties"][difficulty]
            percentage = count / stats["total"] * 100
            print(f"  {difficulty}: {count} ({percentage:.2f}%)")
            
        # 来源-标签交叉分布
        print("\n来源-标签交叉分布:")
        for source in sorted(stats["source_label"].keys()):
            print(f"\n{source}:")
            source_total = sum(stats["source_label"][source].values())
            for label in sorted(stats["source_label"][source].keys()):
                count = stats["source_label"][source][label]
                percentage = count / source_total * 100
                label_name = PipelineType(label).name
                print(f"  {label_name}: {count} ({percentage:.2f}%)")
                
        # 难度-标签交叉分布
        print("\n难度-标签交叉分布:")
        for difficulty in sorted(stats["difficulty_label"].keys()):
            print(f"\n{difficulty}:")
            difficulty_total = sum(stats["difficulty_label"][difficulty].values())
            for label in sorted(stats["difficulty_label"][difficulty].keys()):
                count = stats["difficulty_label"][difficulty][label]
                percentage = count / difficulty_total * 100
                label_name = PipelineType(label).name
                print(f"  {label_name}: {count} ({percentage:.2f}%)")
        
        print("\n" + "="*80)
    
    @staticmethod
    def sample_by_label(data: List[Dict], label: int, sample_size: int) -> List[Dict]:
        """从指定标签的数据中随机采样"""
        labeled_data = [item for item in data if item["label"] == label]
        if len(labeled_data) <= sample_size:
            return labeled_data
        return random.sample(labeled_data, sample_size)
    
    @staticmethod
    def print_samples(samples: List[Dict]):
        """打印采样结果"""
        for i, sample in enumerate(samples, 1):
            print("\n" + "="*80)
            print(f"[Sample {i}/{len(samples)}] Question ID: {sample['question_id']}; Difficulty: {sample['difficulty']}; Pipeline Label: {sample['label']}")
            # print(f"Source: {sample['source']}")
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
    parser = argparse.ArgumentParser(description='Check labeled data distribution and sample examples')
    parser.add_argument('file_path', type=str, help='Path to the labeled data file')
    parser.add_argument('--label', type=int, choices=[1, 2, 3, 4], 
                      help='Label to sample (1: BASIC, 2: INTERMEDIATE, 3: ADVANCED, 4: UNSOLVED)')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of samples to show')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--distribution_only', action='store_true', 
                      help='Only show distribution statistics without samples')
    parser.add_argument('--sample_only', action='store_true',
                      help='Only show samples without distribution statistics')
    
    args = parser.parse_args()
    
    if args.sample_only and args.distribution_only:
        parser.error("Cannot specify both --sample_only and --distribution_only")
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 加载数据
    checker = LabeledDataChecker()
    data = checker.load_labeled_data(args.file_path)
    
    if args.sample_only:
        # 只进行采样
        if args.label is None:
            parser.error("Must specify --label when using --sample_only")
            
        # 计算该标签的总数
        label_count = sum(1 for item in data if item["label"] == args.label)
        label_name = PipelineType(args.label).name
        
        print(f"\nSampling {args.sample_size} examples from {label_name} (Total: {label_count})")
        
        samples = checker.sample_by_label(data, args.label, args.sample_size)
        checker.print_samples(samples)
        
        print(f"\nShowed {len(samples)} samples out of {label_count} {label_name} examples")
    else:
        # 分析并打印分布
        stats = checker.analyze_distribution(data)
        checker.print_distribution(stats)
        
        # 如果不是只看分布且指定了标签，则采样并打印样本
        if not args.distribution_only and args.label is not None:
            label_name = PipelineType(args.label).name
            total_count = stats["labels"][args.label]
            
            print(f"\nSampling {args.sample_size} examples from {label_name} (Total: {total_count})")
            
            samples = checker.sample_by_label(data, args.label, args.sample_size)
            checker.print_samples(samples)
            
            print(f"\nShowed {len(samples)} samples out of {total_count} {label_name} examples")

if __name__ == "__main__":
    main() 