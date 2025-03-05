import os
import torch
import torch.nn as nn
import multiprocessing
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from ..core.config import Config
from datetime import datetime
import json
from pathlib import Path
import yaml
import argparse

# 设置多进程启动方式为spawn
multiprocessing.set_start_method('spawn', force=True)

def setup_distributed():
    """设置分布式训练环境"""
    if 'LOCAL_RANK' not in os.environ:
        return False
    
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    return True

def cleanup_distributed():
    """清理分布式训练环境"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def load_sft_config(config_name: str):
    """加载SFT配置"""
    config_path = Path("config") / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class RoBERTaCascadeTrainer:
    def __init__(self, cascade_dataset: str, pipeline_type: str, model_name: str, sft_config: str):
        self.config = Config()
        self.model_path = self.config.roberta_dir
        self.cascade_dataset_dir = self.config.cascade_data_dir / cascade_dataset
        self.save_dir = self.config.cascade_roberta_save_dir
        self.pipeline_type = pipeline_type  # basic, intermediate, or advanced
        self.model_name = model_name  # 用于保存模型的名称
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.log_file = None
        
        # 加载指定的配置文件
        self.sft_config = load_sft_config(sft_config)
        if self.local_rank == 0:
            print(f"Using SFT config: {sft_config}")
            print(f"Training {pipeline_type} pipeline classifier")
            
    def _log_to_file(self, message: str):
        """写入日志到文件"""
        if self.log_file and self.local_rank == 0:
            with open(self.log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
                
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.model_path,
            padding_side="right"
        )
        
        # 加载二分类模型
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=2,  # 二分类
            problem_type="single_label_classification"
        )
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.sft_config["lora_r"],
            lora_alpha=self.sft_config["lora_alpha"],
            lora_dropout=self.sft_config["lora_dropout"],
            target_modules=self.sft_config["target_modules"],
            bias="none",
        )
        
        # 准备模型
        self.model = get_peft_model(self.model, lora_config)
        
    def prepare_dataset(self):
        """准备数据集"""
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(self.cascade_dataset_dir / f"{self.pipeline_type}_train.json"),
                'validation': str(self.cascade_dataset_dir / f"{self.pipeline_type}_valid.json")
            }
        )
        
        def preprocess_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.sft_config["max_length"],
                return_tensors=None
            )
            # 确保标签是正确的类型和范围
            labels = [int(label) for label in examples["label"]]
            assert all(label in [0, 1] for label in labels), f"Invalid labels found: {labels}"
            tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
            return tokenized
            
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=None
        )
        
        return tokenized_datasets
        
    def compute_metrics(self, eval_pred):
        """计算二分类评估指标"""
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = predictions.argmax(-1)
        labels = eval_pred.label_ids
        
        # 计算基本指标
        total = len(labels)
        correct = (predictions == labels).sum()
        accuracy = correct / total
        
        # 计算精确率、召回率和F1分数
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算预测标签分布
        pred_pos = (predictions == 1).sum()
        pred_neg = (predictions == 0).sum()
        pred_pos_ratio = float(pred_pos) / total
        pred_neg_ratio = float(pred_neg) / total
        
        # 计算实际标签分布
        true_pos = (labels == 1).sum()
        true_neg = (labels == 0).sum()
        true_pos_ratio = float(true_pos) / total
        true_neg_ratio = float(true_neg) / total

        pred_dist_pct = {
            f"pred_pos_ratio": pred_pos_ratio,
            f"pred_neg_ratio": pred_neg_ratio,
        }
        true_dist_pct = {
            f"true_pos_ratio": true_pos_ratio,
            f"true_neg_ratio": true_neg_ratio,
        }

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **pred_dist_pct,
            **true_dist_pct,
        }
        
    def train(self):
        try:
            self.load_model_and_tokenizer()
            tokenized_datasets = self.prepare_dataset()
            
            # 设置checkpoint和训练日志保存路径
            checkpoint_dir = self.save_dir / f"{self.model_name}_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = checkpoint_dir / "training.log"
            
            # 记录训练开始信息
            self._log_to_file(
                f"Training started for {self.pipeline_type} pipeline with:\n"
                f"- Number of training samples: {len(tokenized_datasets['train'])}\n"
                f"- Number of validation samples: {len(tokenized_datasets['validation'])}\n"
            )
            
            # 配置训练参数
            training_args = TrainingArguments(
                output_dir=str(checkpoint_dir),
                per_device_train_batch_size=self.sft_config["per_device_train_batch_size"],
                per_device_eval_batch_size=self.sft_config["per_device_eval_batch_size"],
                gradient_accumulation_steps=self.sft_config["gradient_accumulation_steps"],
                num_train_epochs=self.sft_config["num_train_epochs"],
                learning_rate=self.sft_config["learning_rate"],
                lr_scheduler_type=self.sft_config["lr_scheduler_type"],
                fp16=self.sft_config["fp16"],
                
                # 评估策略配置
                eval_strategy=self.sft_config["eval_strategy"],
                eval_steps=self.sft_config.get("eval_steps", None),
                
                # 保存策略配置
                save_strategy=self.sft_config["save_strategy"],
                save_steps=self.sft_config.get("save_steps", None),
                
                # 日志相关配置
                logging_dir=str(self.config.logs_dir / "cascade" / f"roberta_{self.model_name}"),
                logging_strategy=self.sft_config["logging_strategy"],
                logging_steps=self.sft_config.get("logging_steps", None),
                logging_first_step=self.sft_config["logging_first_step"],
                report_to=["tensorboard"],
                
                # checkpoint相关配置
                load_best_model_at_end=self.sft_config["load_best_model_at_end"],
                metric_for_best_model='precision', # Cascade使用精确率作为主要指标
                greater_is_better=True,
                save_total_limit=self.sft_config["save_total_limit"],
                
                # 其他配置
                ddp_find_unused_parameters=self.sft_config["ddp_find_unused_parameters"],
                local_rank=self.local_rank,
                dataloader_num_workers=self.sft_config["dataloader_num_workers"],
                remove_unused_columns=self.sft_config["remove_unused_columns"],
                warmup_ratio=self.sft_config["warmup_ratio"],
                weight_decay=self.sft_config["weight_decay"],
                max_grad_norm=self.sft_config["max_grad_norm"]
            )
            
            # 创建trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                compute_metrics=self.compute_metrics,
                callbacks=[TrainingCallback(self._log_to_file)]
            )
            
            # 开始训练
            train_result = trainer.train()
            
            # 最终评估
            final_metrics = trainer.evaluate()
            
            # 保存最终模型和结果
            if self.local_rank == 0:
                final_model_dir = self.save_dir / f"final_model_{self.model_name}"
                trainer.save_model(final_model_dir)
                
                # 保存训练结果
                results_file = checkpoint_dir / "training_results.json"
                results_data = {
                    "train_results": train_result.metrics,
                    "eval_results": final_metrics,
                    "train_samples": len(tokenized_datasets["train"]),
                    "eval_samples": len(tokenized_datasets["validation"]),
                    "training_args": training_args.to_dict()
                }
                
                with open(results_file, "w") as f:
                    json.dump(results_data, f, indent=2)
                
        except Exception as e:
            error_message = f"Training error: {str(e)}"
            print(error_message)
            self._log_to_file(f"\nERROR: {error_message}")
            raise e

class TrainingCallback(TrainerCallback):
    """用于记录训练过程的回调"""
    def __init__(self, log_func):
        self.log_func = log_func
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        pred_dist = [metrics.get('eval_pred_pos_ratio', 'N/A'), metrics.get('eval_pred_neg_ratio', 'N/A')]
        true_dist = [metrics.get('eval_true_pos_ratio', 'N/A'), metrics.get('eval_true_neg_ratio', 'N/A')]
        eval_log = (
            f"Evaluation at Step {state.global_step} (epoch {state.epoch:.2f}):\n"
            f"[Accuracy  ] {metrics.get('eval_accuracy', 'N/A'):.5f}\n"
            f"[Precision ] {metrics.get('eval_precision', 'N/A'):.5f}\n"
            f"[Recall    ] {metrics.get('eval_recall', 'N/A'):.5f}\n"
            f"[F1 Score  ] {metrics.get('eval_f1', 'N/A'):.5f}\n"
            f"[Pred Dist ] "
            f"Positive Ratio: {pred_dist[0]*100:.1f}% | "
            f"Negative Ratio: {pred_dist[1]*100:.1f}%\n"
            f"[True Dist ] "
            f"Positive Ratio: {true_dist[0]*100:.1f}% | "
            f"Negative Ratio: {true_dist[1]*100:.1f}%\n"
        )
        self.log_func(eval_log)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft_config', type=str, default='roberta_config',
                       help='Name of the SFT config file under config/ (without .yaml)')
    parser.add_argument('--cascade_dataset', type=str, required=True,
                       help='Name of the cascade dataset directory')
    parser.add_argument('--pipeline_type', type=str, required=True,
                       choices=['basic', 'intermediate', 'advanced'],
                       help='Type of pipeline to train classifier for')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name for saving the model')
    args = parser.parse_args()
    
    is_distributed = False
    try:
        is_distributed = setup_distributed()
        trainer = RoBERTaCascadeTrainer(
            cascade_dataset=args.cascade_dataset,
            pipeline_type=args.pipeline_type,
            model_name=args.model_name,
            sft_config=args.sft_config
        )
        trainer.train()
    finally:
        if is_distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main() 