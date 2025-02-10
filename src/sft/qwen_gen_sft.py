import os
import torch
import multiprocessing
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from accelerate import DistributedDataParallelKwargs
from ..core.config import Config

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

class QwenClassifierTrainer:
    """
    用于分类的Qwen模型 (不添加分类头)
    使用输出的回复进行分类
    """
        
    def __init__(self):
        self.config = Config()
        self.model_path = self.config.model_dir
        self.sft_data_dir = self.config.sft_data_dir
        self.save_dir = self.config.sft_save_dir
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置设备
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={'': self.local_rank}
        )
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        
        # 准备模型
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
    def prepare_dataset(self):
        """准备数据集"""
        # 加载数据集
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(self.sft_data_dir / 'gen_train.json'),
                'validation': str(self.sft_data_dir / 'gen_valid.json')
            }
        )
        
        def tokenize_function(examples):
            texts = [
                f"{prompt}{response}" 
                for prompt, response in zip(examples["prompt"], examples["response"])
            ]
            
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
            
        # 使用单进程处理数据，避免多进程问题
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=None  # 不使用多进程
        )
        
        return tokenized_datasets
        
    def train(self):
        """训练模型"""
        try:
            # 加载模型和数据
            self.load_model_and_tokenizer()
            tokenized_datasets = self.prepare_dataset()
            
            # 配置训练参数
            training_args = TrainingArguments(
                output_dir=self.save_dir,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=2,
                num_train_epochs=3,
                learning_rate=2e-4,
                fp16=True,
                save_steps=100,
                eval_steps=100,
                logging_steps=10,
                eval_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                ddp_find_unused_parameters=False,
                report_to="none",
                local_rank=self.local_rank,
                dataloader_num_workers=0,  # 设置为0，避免数据加载器的多进程问题
                remove_unused_columns=False
            )
            
            # 创建trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            )
            
            # 开始训练
            trainer.train()
            
            # 只在主进程保存模型
            if self.local_rank == 0:
                trainer.save_model(self.save_dir / "final_model_gen")
                
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise e

def main():
    is_distributed = False
    try:
        # 设置分布式环境
        is_distributed = setup_distributed()
        
        # 创建trainer并开始训练
        trainer = QwenClassifierTrainer()
        trainer.train()
        
    finally:
        # 确保在训练结束或发生错误时清理分布式环境
        if is_distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main() 