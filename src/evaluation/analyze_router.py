import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from ..core.utils import load_jsonl
from ..core.config import Config
import argparse

class RouterAnalyzer:
    """分析路由器的预测结果"""
    
    def __init__(self):
        self.config = Config()
        # 标签映射
        self.label_map = {
            "basic": 1,
            "intermediate": 2,
            "advanced": 3
        }
        
    def _extract_prediction(self, router_output: Dict) -> int:
        """从路由器输出中提取预测标签"""
        selected = router_output["output"]["selected_generator"]
        return self.label_map[selected]
    
    def _extract_ground_truth(self, gt_item: Dict) -> int:
        """从ground truth中提取真实标签
        将Unsolved(4)标签映射为Advanced(3)
        """
        label = gt_item["label"]
        # 如果标签是4(Unsolved)，则映射为3(Advanced)
        return min(label, 3)
    
    def analyze(self, router_file: str, ground_truth_file: str):
        """分析路由结果"""
        # 加载数据
        router_results = load_jsonl(router_file)
        ground_truth = load_jsonl(ground_truth_file)
        
        # 构建question_id到ground truth的映射
        gt_map = {item["question_id"]: item for item in ground_truth}
        
        # 收集预测和真实标签
        y_true = []
        y_pred = []
        errors = []  # 记录错误预测的样本
        
        for result in router_results:
            question_id = result["query_id"]
            if question_id not in gt_map:
                continue
                
            gt_item = gt_map[question_id]
            
            try:
                pred_label = self._extract_prediction(result)
                true_label = self._extract_ground_truth(gt_item)
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                
                # 记录错误预测
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
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
        
        # 计算详细指标
        report = classification_report(
            y_true, y_pred,
            labels=[1, 2, 3],
            target_names=["Basic", "Intermediate", "Advanced"],
            digits=4
        )
            
        # 打印分析结果
        print("\n=== Router Analysis Results ===")
        
        # 打印总体指标
        total_samples = len(y_true)
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        overall_accuracy = correct_predictions / total_samples * 100
        
        print(f"\nOverall Statistics:")
        print(f"Total Samples    : {total_samples}")
        print(f"Correct          : {correct_predictions}")
        print(f"Overall Accuracy : {overall_accuracy:.2f}%")
        
        print("\nConfusion Matrix:")
        print("                         Predicted")
        print("True Label      Basic  Intermediate  Advanced    Total")
        print("-------------------------------------------------------")

        # 计算每行的总数
        row_sums = cm.sum(axis=1)

        # 打印每一行
        for i, row in enumerate(cm):
            label = ["Basic", "Intermediate", "Advanced"][i]
            row_total = row_sums[i]
            print(f"{label:12s}  {row[0]:6d}      {row[1]:6d}     {row[2]:6d}     {row_total:4d}")

        # 打印列总数和百分比
        print("-------------------------------------------------------")
        col_sums = cm.sum(axis=0)
        total_sum = col_sums.sum()
        print(f"Total         {col_sums[0]:6d}      {col_sums[1]:6d}     {col_sums[2]:6d}     {total_sum:4d}")
        
        col_percentages = col_sums / total_sum * 100
        print(f"Percentage    {col_percentages[0]:5.1f}%     {col_percentages[1]:5.1f}%    {col_percentages[2]:5.1f}%    100%")

        # 打印每个类别的准确率
        print("\nPer-class Accuracy:")
        for i in range(3):
            accuracy = cm[i,i] / row_sums[i] * 100
            label = ["Basic", "Intermediate", "Advanced"][i]
            print(f"{label:12s}: {accuracy:6.1f}%")

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
    router_file = "results/intermediate_results/20250212_014634/QwenClassifierRouter.jsonl"
    ground_truth = "data/labeled/bird_dev_pipeline_label.jsonl"
    analyzer = RouterAnalyzer()
    analyzer.analyze(router_file, ground_truth)

if __name__ == "__main__":
    main() 