import os
import torch
import multiprocessing
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from accelerate import DistributedDataParallelKwargs
from ..core.config import Config
from typing import Optional, Tuple, Dict
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
import json
from pathlib import Path
from datetime import datetime
import argparse
import yaml
import numpy as np

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

def load_pairwise_config(config_name: str):
    """加载SFT配置"""
    config_path = Path("config") / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class QwenForPairwiseRank(PreTrainedModel):
    """
    用于分类的Qwen模型 (在Qwen的基础上添加分类头)
    使用最后一个token的隐藏状态进行分类
    """
    def __init__(self, base_model, num_labels=3):
        super().__init__(base_model.config)
        self.num_labels = num_labels
        self.qwen = base_model
        # 添加分类头并确保在正确的设备上
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_labels)
        # 将分类头移动到与基础模型相同的设备
        self.classifier.to(base_model.device)
        
    def compute_pairwise_loss(self, logits, chosen_idx, rejected_idx):
        """计算pairwise ranking loss"""
        # 获取每个样本中chosen和rejected pipeline的logits
        chosen_logits = logits[torch.arange(logits.size(0)), chosen_idx]
        rejected_logits = logits[torch.arange(logits.size(0)), rejected_idx]
        
        # 使用margin-based ranking loss
        margin = 1.0
        loss = torch.clamp(rejected_logits - chosen_logits + margin, min=0.0)
        
        return loss.mean()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        # 获取最后一层隐藏状态
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 使用最后一个token的隐藏状态进行分类
        last_hidden_state = outputs.last_hidden_state
        sequence_output = last_hidden_state[:, -1, :]
        # 确保在同一设备上
        sequence_output = sequence_output.to(self.classifier.weight.device)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            if labels.dim() == 2:  # 训练集：pairwise format [batch_size, 2]
                chosen_idx = labels[:, 0]
                rejected_idx = labels[:, 1]
                chosen_idx = chosen_idx.to(logits.device)
                rejected_idx = rejected_idx.to(logits.device)
                loss = self.compute_pairwise_loss(logits, chosen_idx, rejected_idx)
            else:  # 验证集：普通分类 [batch_size]
                labels = labels.to(logits.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def get_classifier_config(self):
        """获取分类头配置"""
        return {
            "num_labels": self.num_labels,
            "hidden_size": self.qwen.config.hidden_size,
            "classifier_dropout": 0.1,  # 如果使用了dropout
            "model_type": "QwenForSequenceClassification"
        }

class QwenPairwiseRankTrainer:
    def __init__(self, pairwise_dataset: str, pairwise_config: str):
        self.config = Config()
        self.model_path = self.config.qwen_dir
        self.pairwise_dataset_dir = self.config.pairwise_data_dir / pairwise_dataset
        self.save_dir = self.config.pairwise_save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.log_file = None
        
        # 加载指定的配置文件
        self.pairwise_config = load_pairwise_config(pairwise_config)
        if self.local_rank == 0:
            print(f"Using SFT config: {pairwise_config}")
        
    def _log_to_file(self, message: str):
        """写入日志到文件"""
        if self.log_file and self.local_rank == 0:  # 只在主进程写日志
            with open(self.log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
                
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={'': self.local_rank}
        )
        
        # 创建分类模型
        self.model = QwenForPairwiseRank(base_model)
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.pairwise_config["lora_r"],
            lora_alpha=self.pairwise_config["lora_alpha"],
            lora_dropout=self.pairwise_config["lora_dropout"],
            target_modules=self.pairwise_config["target_modules"],
            bias="none",
        )
        
        # 准备模型
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
    def prepare_dataset(self):
        """准备pairwise数据集"""
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(self.pairwise_dataset_dir / 'pairwise_train.json'),
                'validation': str(self.pairwise_dataset_dir / 'pairwise_valid.json')
            }
        )
        
        def preprocess_function(examples):
            # 对输入文本进行编码
            tokenized = self.tokenizer(
                examples["input"],
                padding="max_length",
                truncation=True,
                max_length=self.pairwise_config["max_length"],
                return_tensors=None
            )
            
            # 检查是否为验证集样本: 如果是验证集, 则使用original_label作为标签, 否则使用pairwise格式
            is_validation = examples.get("is_validation", [False])[0]
            if is_validation:
                # 验证集：使用original_label作为标签
                labels = torch.tensor([int(label-1) for label in examples["original_label"]], dtype=torch.long)
            else:
                # 训练集：使用pairwise格式
                chosen_idx = [int(idx) for idx in examples["chosen_idx"]]
                rejected_idx = [int(idx) for idx in examples["rejected_idx"]]
                labels = torch.stack([
                    torch.tensor(chosen_idx, dtype=torch.long),
                    torch.tensor(rejected_idx, dtype=torch.long)
                ], dim=1)
            
            tokenized["labels"] = labels
            return tokenized
            
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=None
        )
        
        return tokenized_datasets
        
    def compute_metrics(self, eval_pred):
        """
        为贴近真实使用场景, 使用原始验证集的penalized accuracy作为分类评估方式, 而不使用pairwise ranking accuracy
        """
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        
        # 获取预测的类别
        predicted_classes = np.argmax(predictions, axis=1)
        
        penalty = 0
        total = len(labels)
        penalty_factor = self.pairwise_config["penalty_factor"]
        
        # 计算原始准确率
        correct = (predicted_classes == labels).sum()
        raw_accuracy = correct / total
        
        # 计算每个类别的准确率
        class_accuracies = []
        pred_dist = []
        true_dist = []
        
        for i in range(3):  # 3个类别
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                class_acc = (predicted_classes[class_mask] == labels[class_mask]).mean()
            else:
                class_acc = 0.0
            class_accuracies.append(class_acc)
            
            # 计算预测和真实的分布
            pred_dist.append((predicted_classes == i).mean())
            true_dist.append((labels == i).mean())
        
        # 计算惩罚项：将复杂问题错误分类为简单的惩罚
        for i, pred_class in enumerate(predicted_classes):
            true_class = labels[i]
            if pred_class < true_class and pred_class == 0:
                # 惩罚将Intermediate和Advanced分类到Basic的情况
                penalty += 1
            else:
                penalty += 0
        
        penalty = penalty * penalty_factor
        penalized_accuracy = (correct - penalty) / total
        
        return {
            'accuracy': penalized_accuracy,  # 主要指标
            'raw_accuracy': raw_accuracy,
            'penalty': penalty,
            **{f'class_{i}_accuracy': acc for i, acc in enumerate(class_accuracies)},
            **{f'pred_dist_{i}': dist for i, dist in enumerate(pred_dist)},
            **{f'true_dist_{i}': dist for i, dist in enumerate(true_dist)}
        }
        
    def train(self):
        try:
            self.load_model_and_tokenizer()
            tokenized_datasets = self.prepare_dataset()
            
            # 计算合适的评估步数
            num_train_samples = len(tokenized_datasets["train"])
            # batch_size = 8
            num_gpu = torch.cuda.device_count()
            steps_per_epoch = num_train_samples // (8 * num_gpu)
            
            # 设置checkpoint和训练日志保存路径
            checkpoint_dir = self.save_dir / "qwen_pairwise_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = checkpoint_dir / "training.log"
            
            # 记录训练开始信息
            self._log_to_file(
                f"Training started with:\n"
                f"- Number of GPUs: {num_gpu}\n"
                f"- Number of training samples: {num_train_samples}\n"
                f"- Number of validation samples: {len(tokenized_datasets['validation'])}\n"
                # f"- Steps per epoch: {steps_per_epoch}\n"
            )
            
            # 配置训练参数
            training_args = TrainingArguments(
                output_dir=str(checkpoint_dir),
                # 训练策略配置
                per_device_train_batch_size=self.pairwise_config["per_device_train_batch_size"],
                per_device_eval_batch_size=self.pairwise_config["per_device_eval_batch_size"],
                gradient_accumulation_steps=self.pairwise_config["gradient_accumulation_steps"],
                num_train_epochs=self.pairwise_config["num_train_epochs"],
                learning_rate=self.pairwise_config["learning_rate"],
                lr_scheduler_type=self.pairwise_config["lr_scheduler_type"],
                fp16=self.pairwise_config["fp16"],
                
                # 评估策略配置
                eval_strategy=self.pairwise_config["eval_strategy"],
                eval_steps=self.pairwise_config.get("eval_steps", None),  # 如果是"epoch"模式则不需要
                
                # 保存策略配置
                save_strategy=self.pairwise_config["save_strategy"],
                save_steps=self.pairwise_config.get("save_steps", None),  # 如果是"epoch"模式则不需要
                
                # 日志相关配置
                logging_dir=str(self.config.logs_dir / "pairwise" / "qwen_pairwise"),
                logging_strategy=self.pairwise_config["logging_strategy"],
                logging_steps=self.pairwise_config.get("logging_steps", None),  # 如果是"epoch"模式则不需要
                logging_first_step=self.pairwise_config["logging_first_step"],
                report_to=["tensorboard"],
                
                # checkpoint相关配置
                load_best_model_at_end=self.pairwise_config["load_best_model_at_end"],
                metric_for_best_model=self.pairwise_config["metric_for_best_model"],
                greater_is_better=self.pairwise_config["greater_is_better"],
                save_total_limit=self.pairwise_config["save_total_limit"],
                
                # 其他配置
                ddp_find_unused_parameters=self.pairwise_config["ddp_find_unused_parameters"],
                local_rank=self.local_rank,
                dataloader_num_workers=self.pairwise_config["dataloader_num_workers"],
                remove_unused_columns=self.pairwise_config["remove_unused_columns"],
                warmup_ratio=self.pairwise_config["warmup_ratio"],
                weight_decay=self.pairwise_config["weight_decay"],
                max_grad_norm=self.pairwise_config["max_grad_norm"]
            )
            
            # 创建带日志功能的trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                compute_metrics=self.compute_metrics,
                callbacks=[TrainingCallback(self._log_to_file)]  # 添加回调来记录训练过程
            )
            
            # 开始训练
            train_result = trainer.train()
            
            # 最终评估
            final_metrics = trainer.evaluate()
            
            # 记录最终评估结果
            self._log_to_file("\n\n\nFinal Evaluation Results:")
            for metric_name, value in final_metrics.items():
                self._log_to_file(f"{metric_name}: {value:.4f}")
            
            # 保存最终模型和结果
            if self.local_rank == 0:
                # 保存最终模型
                final_model_dir = self.save_dir / "final_model_classifier"
                trainer.save_model(final_model_dir)
                
                # 保存分类头配置和权重
                classifier_config = self.model.get_classifier_config()
                classifier_state = self.model.classifier.state_dict()
                
                classifier_save = {
                    "config": classifier_config,
                    "state_dict": classifier_state
                }
                
                torch.save(classifier_save, final_model_dir / "classifier.pt")
                
                # 保存训练结果
                results_file = checkpoint_dir / "training_results.json"
                results_data = {
                    "train_results": train_result.metrics,
                    "eval_results": final_metrics,
                    "train_samples": num_train_samples,
                    "eval_samples": len(tokenized_datasets["validation"]),
                    "training_args": training_args.to_dict(),
                    "classifier_config": classifier_config  # 也包含在训练结果中
                }
                
                with open(results_file, "w") as f:
                    json.dump(results_data, f, indent=2)
                
                completion_message = (
                    f"\nTraining completed!\n"
                    f"- Model saved to: {final_model_dir}\n"
                    f"- Results saved to: {results_file}\n"
                    f"- Log saved to: {self.log_file}"
                )
                print(completion_message)
                self._log_to_file(completion_message)
                
        except Exception as e:
            error_message = f"Training error: {str(e)}"
            print(error_message)
            self._log_to_file(f"\nERROR: {error_message}")
            raise e

class TrainingCallback(TrainerCallback):
    """用于记录训练过程的回调"""
    def __init__(self, log_func):
        self.log_func = log_func
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.log_func(f"\n\n<< Epoch {state.epoch:.2f} started >>")
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 提取指标
        penalized_accuracy = metrics.get('eval_accuracy', 'N/A')
        raw_accuracy = metrics.get('eval_raw_accuracy', 'N/A')
        penalty = metrics.get('eval_penalty', 'N/A')
        
        class_accuracies = [
            metrics.get(f'eval_class_{i}_accuracy', 'N/A') 
            for i in range(3)
        ]
        
        pred_dist = [
            metrics.get(f'eval_pred_dist_{i}', 'N/A') 
            for i in range(3)
        ]
        
        true_dist = [
            metrics.get(f'eval_true_dist_{i}', 'N/A') 
            for i in range(3)
        ]
        
        # 构建评估日志
        eval_log = (
            f"Evaluation at Step {state.global_step} (epoch {state.epoch:.2f}):\n"
            f"[Raw Accuracy      ] {raw_accuracy:.5f}\n"
            f"[Penalty          ] {penalty:.5f}\n"
            f"[Penalized Acc    ] {penalized_accuracy:.5f}\n"
            f"[Acc for Classes  ] "
            f"Basic: {class_accuracies[0]:.5f} | "
            f"Inter: {class_accuracies[1]:.5f} | "
            f"Advan: {class_accuracies[2]:.5f}\n"
            f"[Predict Label Dist] "
            f"Basic: {pred_dist[0]:.5f} | Inter: {pred_dist[1]:.5f} | Advan: {pred_dist[2]:.5f}\n"
            f"[Actual Label Dist ] "
            f"Basic: {true_dist[0]:.5f} | Inter: {true_dist[1]:.5f} | Advan: {true_dist[2]:.5f}"
        )
        
        self.log_func(eval_log)
        
    def on_log(self, args, state, control, logs, **kwargs):
        # 记录训练损失等信息
        if "loss" in logs:
            self.log_func(
                f"Step {state.global_step}: "
                f"loss = {logs['loss']:.5f}, "
                f"learning_rate = {logs.get('learning_rate', 'N/A')}"
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairwise_config', type=str, default='pairwise_config',
                       help='Name of the pairwise config file under config/ (without .yaml)')
    parser.add_argument('--pairwise_dataset', type=str, required=True,
                       help='Name of the specified pairwise dataset directory under data/pairwise/')
    args = parser.parse_args()
    
    is_distributed = False
    try:
        is_distributed = setup_distributed()
        trainer = QwenPairwiseRankTrainer(
            pairwise_config=args.pairwise_config,
            pairwise_dataset=args.pairwise_dataset
        )
        trainer.train()
    finally:
        if is_distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main() 