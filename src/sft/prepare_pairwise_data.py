import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from ..core.config import Config
from .instruction_templates import PipelineClassificationTemplates
from ..core.utils import load_jsonl
import argparse

class PairwiseDataProcessor:
    """Process the labeled data to generate the pairwise training dataset"""
    
    def __init__(self, pairwise_dataset: str, seed: int = 42):
        self.config = Config()
        self.pairwise_dataset_dir = self.config.pairwise_data_dir / pairwise_dataset
        self.pairwise_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.templates = PipelineClassificationTemplates()
        self.seed = seed
        
        # Set the random seed
        np.random.seed(self.seed)
        
        # Define the pipeline types and their corresponding indices
        self.pipeline_types = ["basic", "intermediate", "advanced"]
        self.pipeline_to_idx = {
            "basic": 0,
            "intermediate": 1,
            "advanced": 2
        }
        
        # Define the preference pairs according to the label
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
        """Prepare the pairwise training dataset"""
        # Load the data
        data = load_jsonl(labeled_file)
        
        # Construct the pairwise samples
        pairwise_samples = []
        for item in data:
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            
            # Get the label (1-based), label 4 is converted to 3
            label = min(item["label"], 3)
            
            # Get the preference pairs corresponding to the label
            preference_pairs = self.preference_pairs[label]
            
            # Create a sample for each preference pair
            for chosen, rejected in preference_pairs:
                pairwise_samples.append({
                    "question_id": item["question_id"],
                    "input": input_text,
                    "chosen_idx": self.pipeline_to_idx[chosen],
                    "rejected_idx": self.pipeline_to_idx[rejected],
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": item.get("source", "unknown"),
                    "original_label": label
                })
        
        # Shuffle the data
        np.random.shuffle(pairwise_samples)
        
        # Split the training set and validation set
        split_idx = int(len(pairwise_samples) * train_ratio)
        train_samples = pairwise_samples[:split_idx]
        valid_samples = pairwise_samples[split_idx:]
        
        # Save the dataset
        self._save_and_analyze_samples(train_samples, valid_samples)
        
    def _save_and_analyze_samples(self, train_samples: List[Dict], valid_samples: List[Dict]):
        """Save the dataset and analyze the distribution"""
        # Save the dataset
        train_file = self.pairwise_dataset_dir / "pairwise_train.json"
        valid_file = self.pairwise_dataset_dir / "pairwise_valid.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
            
        # Analyze the distribution
        def analyze_distribution(samples: List[Dict]) -> Dict:
            total = len(samples)
            # Original label distribution
            label_dist = {}
            for i in range(1, 5):  # 1-4
                count = sum(1 for s in samples if s["original_label"] == i)
                label_dist[f"Label {i}"] = {
                    "count": count,
                    "percentage": count/total*100
                }
                
            # Preference pair distribution
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