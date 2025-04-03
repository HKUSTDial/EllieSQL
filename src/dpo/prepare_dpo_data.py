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
    """Process the labeled data to generate a fine-tuning dataset 
    (for classification tasks with a classification head model added)
    """
    
    def __init__(self, sft_dataset: str, seed: int = 42):
        self.config = Config()
        # Create the specified dataset directory under sft_data_dir
        self.sft_dataset_dir = self.config.sft_data_dir / sft_dataset
        self.sft_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.templates = PipelineClassificationTemplates()
        self.seed = seed

        self.dpo_dataset_dir = self.config.dpo_data_dir / sft_dataset
        self.dpo_dataset_dir.mkdir(parents=True, exist_ok=True)
        # Set the random seed
        np.random.seed(self.seed)
        
    def prepare_data(self, labeled_file: str, train_ratio: float = 0.8):
        """Prepare the training and validation dataset for the classifier model"""
        # Load data
        data = load_jsonl(labeled_file)
        
        # Construct classification samples
        samples = []
        labels = []  # For stratified sampling
        for item in data:
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            # Convert label 4 to 3, then to 0-based
            label = min(item["label"], 3) - 1
            
            samples.append({
                "text": input_text,
                "label": label,
                "question_id": item["question_id"],
                "source": item.get("source", "unknown")
            })
            labels.append(label)
            
        # Use stratified sampling to split the dataset
        train_indices, valid_indices = self._stratified_split(
            labels, train_ratio=train_ratio
        )
        
        train_samples = [samples[i] for i in train_indices]
        valid_samples = [samples[i] for i in valid_indices]
        
        # Save the dataset
        self._save_and_analyze_samples(train_samples, valid_samples)
        
    def _stratified_split(self, labels: List[int], train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Use stratified sampling to split the dataset"""
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1-train_ratio,
            random_state=self.seed
        )
        
        # Get the split indices
        indices = np.arange(len(labels))
        for train_idx, valid_idx in sss.split(indices, labels):
            return train_idx, valid_idx
            
    def _save_and_analyze_samples(self, train_samples: List[Dict], valid_samples: List[Dict]):
        """Save the dataset and analyze the distribution"""
        # Save the dataset to the specified dataset directory
        train_file = self.dpo_dataset_dir / "classifier_train_pre.json"
        valid_file = self.dpo_dataset_dir / "classifier_valid_pre.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
            
        # Analyze the distribution
        def analyze_distribution(samples: List[Dict]) -> Dict:
            total = len(samples)
            # Label distribution
            label_dist = {}
            for i in range(3):  # 0, 1, 2
                count = sum(1 for s in samples if s["label"] == i)
                label_dist[f"Label {i}"] = {
                    "count": count,
                    "percentage": count/total*100
                }
                
            # Source distribution
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
        
        # Print the analysis results
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
        
        print(f"\nData saved to {self.dpo_dataset_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft_dataset', type=str, required=True,
                       help='Name of the specified SFT dataset directory under data/sft/')
    parser.add_argument('--labeled_file', type=str, required=True,
                       help='Path to the dataset with labels which is to be used for DPO dataset preparation')
    args = parser.parse_args()

    processor = ClassifierDataProcessor(sft_dataset=args.sft_dataset, seed=42)
    processor.prepare_data(args.labeled_file)

if __name__ == "__main__":
    main() 