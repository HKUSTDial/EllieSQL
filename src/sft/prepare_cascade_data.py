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
    """Prepare binary classification data for the cascade classifier"""
    
    def __init__(self, cascade_dataset: str, seed: int = 42):
        self.config = Config()
        self.cascade_dataset_dir = self.config.cascade_data_dir / cascade_dataset
        self.cascade_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.templates = PipelineClassificationTemplates()
        self.seed = seed
        
        # Set the random seed
        np.random.seed(self.seed)
        
    def prepare_data(self, labeled_file: str, balance_config: Dict[str, bool], train_ratio: float = 0.8):
        """Prepare the binary classification data for the three pipelines
        
        :param labeled_file: The path to the labeled data file
        :param balance_config: Configure whether to balance the data for each pipeline, e.g. {"basic": False, "intermediate": True, "advanced": True}
        :param train_ratio: The ratio of the training set
        """
        # Load the data
        data = load_jsonl(labeled_file)
        
        # Prepare the data for each pipeline
        basic_samples = self._prepare_basic_samples(data)
        intermediate_samples = self._prepare_intermediate_samples(data)
        advanced_samples = self._prepare_advanced_samples(data)
        
        # Split and save the data for each pipeline
        self._process_pipeline_data("basic", basic_samples, train_ratio, 
                                  need_balance=balance_config["basic"])
        self._process_pipeline_data("intermediate", intermediate_samples, train_ratio, 
                                  need_balance=balance_config["intermediate"])
        self._process_pipeline_data("advanced", advanced_samples, train_ratio, 
                                  need_balance=balance_config["advanced"])
        
    def _prepare_basic_samples(self, data: List[Dict]) -> List[Dict]:
        """Prepare the binary classification samples for the basic pipeline"""
        samples = []
        for item in data:
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            # The question with basic label is marked as 1, others are marked as 0
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
        """Prepare the binary classification samples for the intermediate pipeline"""
        samples = []
        for item in data:
            # Ignore the questions with basic label
            if item["label"] == 1:
                continue
                
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            # The question with intermediate label is marked as 1, others are marked as 0
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
        """Prepare the binary classification samples for the advanced pipeline"""
        samples = []
        for item in data:
            # Ignore the questions with basic and intermediate labels
            if item["label"] in [1, 2]:
                continue
                
            input_text = self.templates.create_classifier_prompt(
                question=item["question"],
                schema=item["enhanced_linked_schema_wo_info"]
            )
            # The question with advanced label is marked as 1, others are marked as 0
            label = 1 if item["label"] == 3 else 0
            
            samples.append({
                "text": input_text,
                "label": label,
                "question_id": item["question_id"],
                "source": item.get("source", "unknown"),
                "original_label": item["label"]
            })
        return samples
        
    def _balance_samples(self, pipeline: str, samples: List[Dict]) -> List[Dict]:
        """Balance the data through stratified undersampling"""
        # Count the positive and negative samples
        pos_samples = [s for s in samples if s["label"] == 1]
        neg_samples = [s for s in samples if s["label"] == 0]
        pos_count = len(pos_samples)
        neg_count = len(neg_samples)
        
        # Determine the majority and minority classes
        if pos_count > neg_count:
            majority_samples = pos_samples
            minority_samples = neg_samples
            is_pos_majority = True
        else:
            majority_samples = neg_samples
            minority_samples = pos_samples
            is_pos_majority = False
        
        minority_count = len(minority_samples)
        
        # Group the majority class by the original label
        majority_by_original = {}
        for s in majority_samples:
            original_label = s["original_label"]
            if original_label not in majority_by_original:
                majority_by_original[original_label] = []
            majority_by_original[original_label].append(s)
        
        # Calculate the number of samples to sample for each original label
        # Make the total number of majority class samples equal to the number of minority samples, while maintaining the original label ratio
        total_majority = len(majority_samples)
        sampled_majority = []
        for label, samples_of_label in majority_by_original.items():
            # Calculate the number of samples to sample for each original label
            ratio = len(samples_of_label) / total_majority
            sample_size = int(minority_count * ratio)
            if sample_size > 0:  # Ensure at least one sample is sampled
                sampled = np.random.choice(
                    samples_of_label,
                    size=min(sample_size, len(samples_of_label)),
                    replace=False
                ).tolist()
                sampled_majority.extend(sampled)
        
        # Merge samples and shuffle
        balanced_samples = minority_samples + sampled_majority
        np.random.shuffle(balanced_samples)
        
        # Print the distribution information
        print("\n" + "="*50)
        print(f"Balancing details for {pipeline.upper()} pipeline")
        print("="*50)
        majority_label = "positive" if is_pos_majority else "negative"
        minority_label = "negative" if is_pos_majority else "positive"
        print(f"Original {majority_label} samples (majority): {len(majority_samples)}")
        print(f"Original {minority_label} samples (minority): {minority_count}")
        print(f"Original {majority_label} samples by label:")
        for label, samples_of_label in majority_by_original.items():
            print(f"  Label {label}: {len(samples_of_label)}")
        print("After balancing:")
        print(f"  {majority_label} samples: {len(sampled_majority)}")
        print(f"  {minority_label} samples: {len(minority_samples)}")
        for label in majority_by_original.keys():
            count = sum(1 for s in sampled_majority if s["original_label"] == label)
            print(f"    Label {label}: {count}")
        print("="*50)
        return balanced_samples
        
    def _process_pipeline_data(self, pipeline: str, samples: List[Dict], train_ratio: float, need_balance: bool):
        """Process the data for a single pipeline: split the training set and validation set, save and analyze the distribution"""
        # Prepare the labels for stratified sampling
        labels = [s["label"] for s in samples]
        
        # Use stratified sampling to split the dataset
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1-train_ratio,
            random_state=self.seed
        )
        
        # Get the split indices
        indices = np.arange(len(labels))
        for train_idx, valid_idx in sss.split(indices, labels):
            train_samples = [samples[i] for i in train_idx]
            valid_samples = [samples[i] for i in valid_idx]
        
        # If balance is needed, only balance the training set
        if need_balance == True:
            train_samples = self._balance_samples(pipeline, train_samples)
            
        # Save the dataset
        train_file = self.cascade_dataset_dir / f"{pipeline}_train.json"
        valid_file = self.cascade_dataset_dir / f"{pipeline}_valid.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
            
        # Analyze and print the distribution
        if need_balance == False:
            print("="*50)
        print(f"{pipeline.upper()} Pipeline Train/Val Data Distribution")
        print("="*50)
        print(f"Balanced?: {need_balance}")
        print(f"Total samples: {len(samples)}")
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(valid_samples)}")
        
        # Analyze and print the label distribution
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
        
        print("Label Distribution:")
        print("Training set:")
        print(f"  Positive: {train_dist['positive']} ({train_dist['pos_ratio']:.1f}%)")
        print(f"  Negative: {train_dist['negative']} ({100-train_dist['pos_ratio']:.1f}%)")
        print("Validation set:")
        print(f"  Positive: {valid_dist['positive']} ({valid_dist['pos_ratio']:.1f}%)")
        print(f"  Negative: {valid_dist['negative']} ({100-valid_dist['pos_ratio']:.1f}%)")
        print("="*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cascade_dataset', type=str, required=True,
                       help='Name of the specified cascade dataset directory under data/cascade/')
    parser.add_argument('--labeled_file', type=str, required=True,
                       help='Path to the labeled data file')
    # Add data balance options, use type=lambda x: x.lower() == 'true' to convert to a boolean value
    parser.add_argument('--basic_balance', type=lambda x: x.lower() == 'true', default=False,
                       choices=[True, False],
                       help='Whether to balance basic pipeline data (True/False)')
    parser.add_argument('--intermediate_balance', type=lambda x: x.lower() == 'true', default=False,
                       choices=[True, False],
                       help='Whether to balance intermediate pipeline data (True/False)')
    parser.add_argument('--advanced_balance', type=lambda x: x.lower() == 'true', default=False,
                       choices=[True, False],
                       help='Whether to balance advanced pipeline data (True/False)')
    args = parser.parse_args()
    
    # Build the balance configuration
    balance_config = {
        "basic": args.basic_balance,
        "intermediate": args.intermediate_balance,
        "advanced": args.advanced_balance
    }
    
    processor = CascadeDataProcessor(cascade_dataset=args.cascade_dataset, seed=42)
    processor.prepare_data(args.labeled_file, balance_config)

if __name__ == "__main__":
    main() 