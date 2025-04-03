import json
import random
from pathlib import Path
from typing import List, Dict
import argparse
from collections import defaultdict
from ..core.utils import load_jsonl
from .labeler import PipelineType

class LabeledDataChecker:
    """Check the distribution of labeled data and sample examples"""
    
    @staticmethod
    def load_labeled_data(file_path: str) -> List[Dict]:
        """Load the labeled data"""
        return load_jsonl(file_path)
    
    @staticmethod
    def analyze_distribution(data: List[Dict]) -> Dict:
        """Analyze the data distribution"""
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
            
            # Basic statistics
            stats["labels"][label] += 1
            stats["sources"][source] += 1
            stats["difficulties"][difficulty] += 1
            
            # Cross-statistics
            stats["source_label"][source][label] += 1
            stats["difficulty_label"][difficulty][label] += 1
            
        return stats
    
    @staticmethod
    def print_distribution(stats: Dict):
        """Print the distribution statistics"""
        print("\n" + "="*80)
        print("Data Distribution Statistics")
        print("="*80)
        
        # Total statistics
        print(f"\nTotal number of samples: {stats['total']}")
        
        # Label distribution
        print("\nLabel Distribution:")
        for label in sorted(stats["labels"].keys()):
            count = stats["labels"][label]
            percentage = count / stats["total"] * 100
            label_name = PipelineType(label).name
            print(f"  {label_name}: {count} ({percentage:.2f}%)")
            
        # Source distribution
        print("\nSource Distribution:")
        for source in sorted(stats["sources"].keys()):
            count = stats["sources"][source]
            percentage = count / stats["total"] * 100
            print(f"  {source}: {count} ({percentage:.2f}%)")
            
        # Difficulty distribution
        print("\nDifficulty Distribution:")
        for difficulty in sorted(stats["difficulties"].keys()):
            count = stats["difficulties"][difficulty]
            percentage = count / stats["total"] * 100
            print(f"  {difficulty}: {count} ({percentage:.2f}%)")
            
        # Source-label cross-distribution
        print("\nSource-label cross-distribution:")
        for source in sorted(stats["source_label"].keys()):
            print(f"\n{source}:")
            source_total = sum(stats["source_label"][source].values())
            for label in sorted(stats["source_label"][source].keys()):
                count = stats["source_label"][source][label]
                percentage = count / source_total * 100
                label_name = PipelineType(label).name
                print(f"  {label_name}: {count} ({percentage:.2f}%)")
                
        # Difficulty-label cross-distribution
        print("\nDifficulty-label cross-distribution:")
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
        """Sample from the data of the specified label"""
        labeled_data = [item for item in data if item["label"] == label]
        if len(labeled_data) <= sample_size:
            return labeled_data
        return random.sample(labeled_data, sample_size)
    
    @staticmethod
    def print_samples(samples: List[Dict]):
        """Print the sampling results"""
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
    
    # Set the random seed
    random.seed(args.seed)
    
    # Load data
    checker = LabeledDataChecker()
    data = checker.load_labeled_data(args.file_path)
    
    if args.sample_only:
        # Only sample
        if args.label is None:
            parser.error("Must specify --label when using --sample_only")
            
        # Calculate the total number of the specified label
        label_count = sum(1 for item in data if item["label"] == args.label)
        label_name = PipelineType(args.label).name
        
        print(f"\nSampling {args.sample_size} examples from {label_name} (Total: {label_count})")
        
        samples = checker.sample_by_label(data, args.label, args.sample_size)
        checker.print_samples(samples)
        
        print(f"\nShowed {len(samples)} samples out of {label_count} {label_name} examples")
    else:
        # Analyze and print the distribution
        stats = checker.analyze_distribution(data)
        checker.print_distribution(stats)
        
        # If not only showing distribution and specified label, then sample and print samples
        if not args.distribution_only and args.label is not None:
            label_name = PipelineType(args.label).name
            total_count = stats["labels"][args.label]
            
            print(f"\nSampling {args.sample_size} examples from {label_name} (Total: {total_count})")
            
            samples = checker.sample_by_label(data, args.label, args.sample_size)
            checker.print_samples(samples)
            
            print(f"\nShowed {len(samples)} samples out of {total_count} {label_name} examples")

if __name__ == "__main__":
    main() 