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
    PreTrainedModel
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

class QwenClassifierTrainer:
    def __init__(self):
        self.config = Config()
        self.model_path = self.config.model_dir
        self.finetune_data_dir = self.config.finetune_data_dir
        self.save_dir = self.config.finetune_save_dir
        
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
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(self.finetune_data_dir / 'classifier_train.json'),
                'validation': str(self.finetune_data_dir / 'classifier_valid.json')
            }
        )
        
        def preprocess_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors=None
            )
            # 确保标签是正确的类型和范围
            labels = [int(label) for label in examples["label"]]  # 已经是0-based的标签
            # 验证标签范围
            assert all(0 <= label < 3 for label in labels), f"Invalid label found in {labels}"
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
        """计算评估指标"""
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = predictions.argmax(-1)
        labels = eval_pred.label_ids
        
        correct = (predictions == labels).sum()
        total = len(labels)
        accuracy = float(correct) / total
        
        return {"accuracy": accuracy}
        
    def train(self):
        """训练模型"""
        try:
            self.load_model_and_tokenizer()
            tokenized_datasets = self.prepare_dataset()
            
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
                metric_for_best_model="accuracy",
                ddp_find_unused_parameters=False,
                report_to="none",
                local_rank=self.local_rank,
                dataloader_num_workers=0,
                remove_unused_columns=False
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                compute_metrics=self.compute_metrics
            )
            
            trainer.train()
            
            if self.local_rank == 0:
                trainer.save_model(self.save_dir / "final_model_classifier")
                
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise e

def main():
    is_distributed = False
    try:
        is_distributed = setup_distributed()
        trainer = QwenClassifierTrainer()
        trainer.train()
    finally:
        if is_distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main() 