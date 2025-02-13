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

class QwenForSequenceClassification(PreTrainedModel):
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
            # 确保标签在正确的设备上
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

class QwenClassifierTrainer:
    def __init__(self, sft_dataset: str, sft_config: str):
        self.config = Config()
        self.model_path = self.config.model_dir
        self.sft_dataset_dir = self.config.sft_data_dir / sft_dataset
        self.save_dir = self.config.sft_save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.log_file = None
        
        # 加载指定的配置文件
        self.sft_config = load_sft_config(sft_config)
        if self.local_rank == 0:
            print(f"Using SFT config: {sft_config}")
        
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
        self.model = QwenForSequenceClassification(base_model)
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.sft_config["lora_r"],
            lora_alpha=self.sft_config["lora_alpha"],
            lora_dropout=self.sft_config["lora_dropout"],
            target_modules=self.sft_config["target_modules"],
            bias="none",
        )
        
        # 准备模型
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
    def prepare_dataset(self):
        """准备数据集"""
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(self.sft_dataset_dir / 'classifier_train.json'),
                'validation': str(self.sft_dataset_dir / 'classifier_valid.json')
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
            # 验证标签范围并打印异常值
            for i, label in enumerate(labels):
                if not (0 <= label < 3):
                    print(f"Warning: Invalid label {label} at index {i}")
            assert all(0 <= label < 3 for label in labels), f"Invalid labels found: {labels}"
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
        """计算评估指标, 对于将复杂问题错误分类到简单pipeline的情况增加惩罚"""
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = predictions.argmax(-1)  # 转换为0-based标签
        labels = eval_pred.label_ids  # 0-based标签
        
        total = len(labels)
        correct = 0
        penalty = 0
        penalty_factor = 0.1 # 惩罚因子, 如为0则不惩罚
        
        # 计算每个类别的统计信息
        class_correct = {i: 0 for i in range(3)}
        class_total = {i: 0 for i in range(3)}
        pred_dist = {i: 0 for i in range(3)}
        
        for pred, label in zip(predictions, labels):
            class_total[label] += 1
            pred_dist[pred] += 1
            
            if pred == label:
                # 正确分类
                correct += 1
                class_correct[label] += 1
            elif label > pred and pred == 0:
                # 惩罚将Intermediate和Advanced分类到Basic的情况
                penalty += 1 * penalty_factor
                # print(f"******** label: {label}, pred: {pred} **********")
        
        # 计算带惩罚的准确率
        penalized_accuracy = (correct - penalty) / total
        
        # 计算每个类别的准确率
        class_accuracy = {}
        for i in range(3):
            if class_total[i] > 0:
                class_accuracy[f"class_{i}_accuracy"] = float(class_correct[i]) / class_total[i]
            else:
                class_accuracy[f"class_{i}_accuracy"] = 0.0
        
        # 计算分布百分比
        pred_dist_pct = {
            f"pred_dist_{i}": float(count) / total 
            for i, count in pred_dist.items()
        }
        true_dist_pct = {
            f"true_dist_{i}": float(class_total[i]) / total 
            for i in range(3)
        }
        
        # 记录惩罚信息
        penalty_info = {
            "penalty": float(penalty),
            "raw_accuracy": float(correct) / total,
            "penalized_accuracy": float(penalized_accuracy)
        }
        
        # 返回所有指标
        return {
            "accuracy": penalized_accuracy,  # 使用带惩罚的准确率
            **class_accuracy,
            **pred_dist_pct,
            **true_dist_pct,
            **penalty_info
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
            checkpoint_dir = self.save_dir / "qwen_classifier_checkpoints"
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
                per_device_train_batch_size=self.sft_config["per_device_train_batch_size"],
                per_device_eval_batch_size=self.sft_config["per_device_eval_batch_size"],
                gradient_accumulation_steps=self.sft_config["gradient_accumulation_steps"],
                num_train_epochs=self.sft_config["num_train_epochs"],
                learning_rate=self.sft_config["learning_rate"],
                lr_scheduler_type=self.sft_config["lr_scheduler_type"],
                fp16=self.sft_config["fp16"],
                
                # 评估策略配置
                eval_strategy=self.sft_config["eval_strategy"],
                eval_steps=self.sft_config.get("eval_steps", None),  # 如果是"epoch"模式则不需要
                
                # 保存策略配置
                save_strategy=self.sft_config["save_strategy"],
                save_steps=self.sft_config.get("save_steps", None),  # 如果是"epoch"模式则不需要
                
                # 日志相关配置
                logging_dir=str(self.config.logs_dir / "sft" / "qwen_classifier"),
                logging_strategy=self.sft_config["logging_strategy"],
                logging_steps=self.sft_config.get("logging_steps", None),  # 如果是"epoch"模式则不需要
                logging_first_step=self.sft_config["logging_first_step"],
                report_to=["tensorboard"],
                
                # checkpoint相关配置
                load_best_model_at_end=self.sft_config["load_best_model_at_end"],
                metric_for_best_model=self.sft_config["metric_for_best_model"],
                greater_is_better=self.sft_config["greater_is_better"],
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
    parser.add_argument('--sft_config', type=str, default='sft_config',
                       help='Name of the SFT config file under config/ (without .yaml)')
    parser.add_argument('--sft_dataset', type=str, required=True,
                       help='Name of the specified SFT dataset directory under data/sft/')
    args = parser.parse_args()
    
    is_distributed = False
    try:
        is_distributed = setup_distributed()
        trainer = QwenClassifierTrainer(
            sft_config=args.sft_config,
            sft_dataset=args.sft_dataset
        )
        trainer.train()
    finally:
        if is_distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main() 