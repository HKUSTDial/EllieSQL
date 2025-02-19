import os
import torch
import multiprocessing
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from trl import DPOTrainer, DPOConfig
from accelerate import DistributedDataParallelKwargs
from ..core.config import Config
from typing import Optional, Tuple, Dict
import torch.distributed as dist
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

def load_dpo_config(config_name: str):
    """加载DPO配置"""
    config_path = Path("config") / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class QwenDPOTrainer:
    def __init__(self, dpo_dataset: str, dpo_config: str):
        self.config = Config()
        self.model_path = self.config.qwen_dir
        self.dpo_dataset_dir = self.config.pairwise_data_dir / dpo_dataset
        self.save_dir = self.config.dpo_save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.log_file = None
        
        # 加载指定的配置文件
        self.dpo_config = load_dpo_config(dpo_config)
        if self.local_rank == 0:
            print(f"Using DPO config: {dpo_config}")
            
    def _log_to_file(self, message: str):
        """写入日志到文件"""
        if self.log_file and self.local_rank == 0:  # 只在主进程写日志
            with open(self.log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
                
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={'': self.local_rank}
        )
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.dpo_config["lora_r"],
            lora_alpha=self.dpo_config["lora_alpha"],
            lora_dropout=self.dpo_config["lora_dropout"],
            target_modules=self.dpo_config["target_modules"],
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
                'train': str(self.dpo_dataset_dir / 'dpo_train.json'),
                'validation': str(self.dpo_dataset_dir / 'dpo_valid.json')
            }
        )
        
        # 添加数据预处理
        def preprocess_function(examples):
            # 处理输入文本
            prompts = examples["input"]
            
            # 编码输入
            prompt_tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.dpo_config["max_prompt_length"],
                return_tensors=None
            )
            
            # 编码chosen和rejected回答
            chosen_tokens = self.tokenizer(
                examples["chosen"],
                padding=True,
                truncation=True,
                max_length=self.dpo_config["max_length"],
                return_tensors=None
            )
            
            rejected_tokens = self.tokenizer(
                examples["rejected"],
                padding=True,
                truncation=True,
                max_length=self.dpo_config["max_length"],
                return_tensors=None
            )
            
            # 确保所有token_ids都是整数类型
            for tokens in [prompt_tokens, chosen_tokens, rejected_tokens]:
                tokens["input_ids"] = [
                    [int(x) for x in seq] for seq in tokens["input_ids"]
                ]
                tokens["attention_mask"] = [
                    [int(x) for x in seq] for seq in tokens["attention_mask"]
                ]
            
            return {
                "prompt": prompts,
                "chosen": examples["chosen"],
                "rejected": examples["rejected"],
                "prompt_ids": prompt_tokens["input_ids"],
                "prompt_attention_mask": prompt_tokens["attention_mask"],
                "chosen_ids": chosen_tokens["input_ids"],
                "chosen_attention_mask": chosen_tokens["attention_mask"],
                "rejected_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
            }
        
        # 应用预处理
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing datasets..."
        )
        
        return processed_datasets
        
    def train(self):
        try:
            self.load_model_and_tokenizer()
            dataset = self.prepare_dataset()
            
            # 设置checkpoint和训练日志保存路径
            checkpoint_dir = self.save_dir / "qwen_dpo_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = checkpoint_dir / "training.log"
            
            # 记录训练开始信息
            self._log_to_file(
                f"Training started with:\n"
                f"- Number of training samples: {len(dataset['train'])}\n"
                f"- Number of validation samples: {len(dataset['validation'])}\n"
            )
            
            # 配置DPO训练参数
            dpo_config = DPOConfig(
                output_dir=str(checkpoint_dir),
                # 基本训练参数
                per_device_train_batch_size=self.dpo_config["per_device_train_batch_size"],
                per_device_eval_batch_size=self.dpo_config["per_device_eval_batch_size"],
                gradient_accumulation_steps=self.dpo_config["gradient_accumulation_steps"],
                num_train_epochs=self.dpo_config["num_train_epochs"],
                learning_rate=self.dpo_config["learning_rate"],
                lr_scheduler_type=self.dpo_config["lr_scheduler_type"],
                fp16=self.dpo_config["fp16"],
                bf16=self.dpo_config["bf16"],

                # DPO参数
                beta=self.dpo_config["beta"],
                max_prompt_length=self.dpo_config["max_prompt_length"],
                max_length=self.dpo_config["max_length"],
                
                # 评估策略配置
                eval_strategy=self.dpo_config["eval_strategy"],
                eval_steps=self.dpo_config.get("eval_steps", None),

                # 保存策略配置
                save_strategy=self.dpo_config["save_strategy"],
                save_steps=self.dpo_config.get("save_steps", None),

                # 日志相关配置
                logging_dir=str(self.config.logs_dir / "dpo" / "qwen_dpo"),
                logging_strategy=self.dpo_config["logging_strategy"],
                logging_steps=self.dpo_config.get("logging_steps", None),
                logging_first_step=self.dpo_config["logging_first_step"],
                report_to=["tensorboard"],
                
                # checkpoint相关配置
                load_best_model_at_end=self.dpo_config["load_best_model_at_end"],
                save_total_limit=self.dpo_config["save_total_limit"],
                
                # 其他配置
                local_rank=self.local_rank,
                ddp_find_unused_parameters=self.dpo_config["ddp_find_unused_parameters"],
                remove_unused_columns=self.dpo_config["remove_unused_columns"],
                warmup_ratio=self.dpo_config["warmup_ratio"],
                weight_decay=self.dpo_config["weight_decay"],
                max_grad_norm=self.dpo_config["max_grad_norm"],
            )
            
            # 创建DPO trainer
            trainer = DPOTrainer(
                model=self.model,
                ref_model=None,  # 使用同一个模型作为参考
                args=dpo_config,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                processing_class=self.tokenizer
            )
            
            # 开始训练
            train_result = trainer.train()
            
            # 最终评估
            final_metrics = trainer.evaluate()
            
            # 记录最终评估结果
            self._log_to_file("\nFinal Evaluation Results:")
            for metric_name, value in final_metrics.items():
                self._log_to_file(f"{metric_name}: {value:.4f}")
            
            # 保存最终模型和结果
            if self.local_rank == 0:
                # 保存最终模型
                final_model_dir = self.save_dir / "final_model_dpo"
                trainer.save_model(final_model_dir)
                
                # 保存训练结果
                results_file = checkpoint_dir / "training_results.json"
                results_data = {
                    "train_results": train_result.metrics,
                    "eval_results": final_metrics,
                    "train_samples": len(dataset["train"]),
                    "eval_samples": len(dataset["validation"]),
                    "dpo_config": self.dpo_config
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo_config', type=str, default='dpo_config',
                       help='Name of the DPO config file under config/ (without .yaml)')
    parser.add_argument('--dpo_dataset', type=str, required=True,
                       help='Name of the specified DPO dataset directory under data/dpo/')
    args = parser.parse_args()
    
    is_distributed = False
    try:
        is_distributed = setup_distributed()
        trainer = QwenDPOTrainer(
            dpo_config=args.dpo_config,
            dpo_dataset=args.dpo_dataset
        )
        trainer.train()
    finally:
        if is_distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main() 