import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from ..core.utils import load_jsonl
from ..core.config import Config
import argparse

class RouterAnalyzer:
    """Analyze the prediction results of the router"""
    
    def __init__(self):
        self.config = Config()
        # Label mapping
        self.label_map = {
            "basic": 1,
            "intermediate": 2,
            "advanced": 3
        }
        
    def _extract_prediction(self, router_output: Dict) -> int:
        """Extract the predicted label from the router output"""
        selected = router_output["output"]["selected_generator"]
        return self.label_map[selected]
    
    def _extract_oracle_label(self, gt_item: Dict) -> int:
        """Extract the oracle label
        Map Unsolved(4) to Advanced(3)
        """
        label = gt_item["label"]
        # If the label is 4(Unsolved), map it to 3(Advanced)
        return min(label, 3)
    
    def analyze(self, router_file: str, oracle_file: str):
        """Analyze the routing results"""
        # Load data
        router_results = load_jsonl(router_file)
        oracle_labels = load_jsonl(oracle_file)
        
        # Build the mapping from question_id to oracle label
        gt_map = {item["question_id"]: item for item in oracle_labels}
        
        # Collect predicted and true labels
        y_true = []
        y_pred = []
        errors = []  # Record the samples with incorrect predictions
        
        for result in router_results:
            question_id = result["query_id"]
            if question_id not in gt_map:
                continue
                
            gt_item = gt_map[question_id]
            
            try:
                pred_label = self._extract_prediction(result)
                true_label = self._extract_oracle_label(gt_item)
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                
                # Record the samples with incorrect predictions
                if pred_label != true_label:
                    errors.append({
                        "question_id": question_id,
                        "question": gt_item["question"],
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "source": gt_item.get("source", "unknown"),
                        "difficulty": gt_item.get("difficulty", "unknown")
                    })
                    
            except Exception as e:
                print(f"Error processing question {question_id}: {str(e)}")
                continue
        
        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
        
        # Calculate the sum of the main diagonal and above (overestimation)
        upper_sum = 0
        for i in range(3):
            for j in range(i, 3):  # Only calculate the diagonal and above
                upper_sum += cm[i, j]
        
        # Calculate the sum of the main diagonal and below (underestimation)
        lower_sum = 0
        for i in range(3):
            for j in range(i + 1):  # Only calculate the diagonal and below
                lower_sum += cm[i, j]
        
        # Calculate the total number of samples
        total_samples = cm.sum()
        
        # Calculate the detailed metrics
        report = classification_report(
            y_true, y_pred,
            labels=[1, 2, 3],
            target_names=["Basic", "Intermediate", "Advanced"],
            digits=4
        )
            
        # Print the analysis results
        print("\n=== Router Analysis Results ===")
        
        # Print the overall metrics
        total_samples = len(y_true)
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        overall_agreement = correct_predictions / total_samples * 100
        
        print(f"\nOverall Statistics:")
        print(f"Total Samples     : {total_samples}")
        print(f"Match with Oracle : {correct_predictions}")
        print(f"Overall Agreement : {overall_agreement:.2f}%")
        
        print("\nConfusion Matrix:")
        print("                         Predicted")
        print("True Label      Basic  Intermediate  Advanced    Total")
        print("-------------------------------------------------------")

        # Calculate the sum of each row
        row_sums = cm.sum(axis=1)

        # Print each row
        for i, row in enumerate(cm):
            label = ["Basic", "Intermediate", "Advanced"][i]
            row_total = row_sums[i]
            print(f"{label:12s}  {row[0]:6d}      {row[1]:6d}     {row[2]:6d}     {row_total:4d}")

        print("-------------------------------------------------------")
        col_sums = cm.sum(axis=0)
        total_sum = col_sums.sum()
        print(f"Total         {col_sums[0]:6d}      {col_sums[1]:6d}     {col_sums[2]:6d}     {total_sum:4d}")
        
        col_percentages = col_sums / total_sum * 100
        print(f"Percentage    {col_percentages[0]:5.1f}%     {col_percentages[1]:5.1f}%    {col_percentages[2]:5.1f}%    100%")

        # Print the analysis of the main diagonal and below
        print("\nDiagonal Analysis:")
        print(f"Upper Triangle (Overestimation):")
        print(f"  Sum: {upper_sum}")
        print(f"  Percentage: {upper_sum/total_samples*100:.2f}%")
        print(f"Lower Triangle (Underestimation):")
        print(f"  Sum: {lower_sum}")
        print(f"  Percentage: {lower_sum/total_samples*100:.2f}%")

        # Print the accuracy of each class
        print("\nPer-class Agreement:")
        for i in range(3):
            agreement = cm[i,i] / row_sums[i] * 100
            label = ["Basic", "Intermediate", "Advanced"][i]
            print(f"{label:12s}: {agreement:6.1f}%")

        print("\nClassification Report:")
        print(report)
        
        print("\nError Analysis:")
        error_by_source = {}
        error_by_difficulty = {}
        
        for error in errors:
            source = error["source"]
            difficulty = error["difficulty"]
            
            if source not in error_by_source:
                error_by_source[source] = []
            error_by_source[source].append(error)
            
            if difficulty not in error_by_difficulty:
                error_by_difficulty[difficulty] = []
            error_by_difficulty[difficulty].append(error)
            
        print("\nErrors by Source:")
        for source, errs in error_by_source.items():
            print(f"{source}: {len(errs)} errors")
            
        print("\nErrors by Difficulty:")
        for diff, errs in error_by_difficulty.items():
            print(f"{diff}: {len(errs)} errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--router_file', type=str, required=True,
                       help='Path to router output file')
    parser.add_argument('--oracle_file', type=str, required=True,
                       help='Path to oracle file containing correct labels')
    args = parser.parse_args()
    
    analyzer = RouterAnalyzer()
    analyzer.analyze(args.router_file, args.oracle_file)

if __name__ == "__main__":
    main() 