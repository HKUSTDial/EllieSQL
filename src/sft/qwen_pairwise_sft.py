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

# Set the multiprocessing start method to spawn
multiprocessing.set_start_method('spawn', force=True)

def setup_distributed():
    """Set up the distributed training environment"""
    if 'LOCAL_RANK' not in os.environ:
        return False
    
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    return True

def cleanup_distributed():
    """Clean up the distributed training environment"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def load_pairwise_config(config_name: str):
    """Load SFT configuration"""
    config_path = Path("config") / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class QwenForPairwiseRank(PreTrainedModel):
    """
    Qwen model for classification (adds a classification head on top of Qwen)
    Uses the last token's hidden state for classification
    """
    def __init__(self, base_model, num_labels=3):
        super().__init__(base_model.config)
        self.num_labels = num_labels
        self.qwen = base_model
        # Add classification head and ensure it's on the correct device
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_labels)
        # Move the classification head to the same device as the base model
        self.classifier.to(base_model.device)
        
    def compute_pairwise_loss(self, logits, chosen_idx, rejected_idx):
        """Compute pairwise ranking loss"""
        # Get the logits for the chosen and rejected pipelines for each sample
        chosen_logits = logits[torch.arange(logits.size(0)), chosen_idx]
        rejected_logits = logits[torch.arange(logits.size(0)), rejected_idx]
        
        # Use margin-based ranking loss
        margin = 1.0
        loss = torch.clamp(rejected_logits - chosen_logits + margin, min=0.0)
        
        return loss.mean()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # labels is now a [batch_size, 2] tensor
        **kwargs
    ) -> SequenceClassifierOutput:
        # Get the last hidden state
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Use the last hidden state for classification
        last_hidden_state = outputs.last_hidden_state
        sequence_output = last_hidden_state[:, -1, :]
        # Ensure on the same device
        sequence_output = sequence_output.to(self.classifier.weight.device)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # Extract chosen_idx and rejected_idx from labels
            chosen_idx = labels[:, 0]  # [batch_size]
            rejected_idx = labels[:, 1]  # [batch_size]
            
            # Compute pairwise ranking loss
            chosen_idx = chosen_idx.to(logits.device)
            rejected_idx = rejected_idx.to(logits.device)
            loss = self.compute_pairwise_loss(logits, chosen_idx, rejected_idx)
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def get_classifier_config(self):
        """Get the classifier configuration"""
        return {
            "num_labels": self.num_labels,
            "hidden_size": self.qwen.config.hidden_size,
            "classifier_dropout": 0.1,  # If dropout is used
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
        
        # Load the specified configuration file
        self.pairwise_config = load_pairwise_config(pairwise_config)
        if self.local_rank == 0:
            print(f"Using SFT config: {pairwise_config}")
        
    def _log_to_file(self, message: str):
        """Write to the log file"""
        if self.log_file and self.local_rank == 0:  # Only write logs in the main process
            with open(self.log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
                
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the base model
        base_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={'': self.local_rank}
        )
        
        # Create the classification model
        self.model = QwenForPairwiseRank(base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.pairwise_config["lora_r"],
            lora_alpha=self.pairwise_config["lora_alpha"],
            lora_dropout=self.pairwise_config["lora_dropout"],
            target_modules=self.pairwise_config["target_modules"],
            bias="none",
        )
        
        # Prepare the model
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
    def prepare_dataset(self):
        """Prepare the pairwise dataset"""
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(self.pairwise_dataset_dir / 'pairwise_train.json'),
                'validation': str(self.pairwise_dataset_dir / 'pairwise_valid.json')
            }
        )
        
        def preprocess_function(examples):
            # Encode the input text
            tokenized = self.tokenizer(
                examples["input"],  # Note here we use "input" instead of "text"
                padding="max_length",
                truncation=True,
                max_length=self.pairwise_config["max_length"],
                return_tensors=None
            )
            
            # Add chosen_idx and rejected_idx
            chosen_idx = [int(idx) for idx in examples["chosen_idx"]]
            rejected_idx = [int(idx) for idx in examples["rejected_idx"]]
            
            # Verify index range
            for i, (chosen, rejected) in enumerate(zip(chosen_idx, rejected_idx)):
                if not (0 <= chosen < 3 and 0 <= rejected < 3):
                    print(f"Warning: Invalid index at position {i}: chosen={chosen}, rejected={rejected}")
            assert all(0 <= idx < 3 for idx in chosen_idx + rejected_idx), "Invalid indices found"
            
            # Convert indices to tensors
            tokenized["chosen_idx"] = torch.tensor(chosen_idx, dtype=torch.long)
            tokenized["rejected_idx"] = torch.tensor(rejected_idx, dtype=torch.long)
            
            # Combine chosen_idx and rejected_idx into a labels tensor
            labels = torch.stack([
                tokenized["chosen_idx"],
                tokenized["rejected_idx"]
            ], dim=1)  # shape: [batch_size, 2]
            
            # Use the labels field, as this is the default field Trainer looks for
            tokenized["labels"] = labels
            
            # Delete the original idx fields to avoid conflicts
            del tokenized["chosen_idx"]
            del tokenized["rejected_idx"]
            
            return tokenized
            
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=None
        )
        
        return tokenized_datasets
        
    def compute_metrics(self, eval_pred):
        """Compute the evaluation metrics for pairwise ranking"""
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids  # Here it contains chosen_idx and rejected_idx
        
        # Separate chosen_idx and rejected_idx from labels
        chosen_idx = labels[:, 0]  # Assume the first column is chosen_idx
        rejected_idx = labels[:, 1]  # Assume the second column is rejected_idx
        
        # Calculate the predicted scores for each sample
        batch_size = logits.shape[0]
        chosen_scores = logits[np.arange(batch_size), chosen_idx]
        rejected_scores = logits[np.arange(batch_size), rejected_idx]
        
        # Calculate the correct ranking ratio (chosen_score > rejected_score)
        correct_ranking = (chosen_scores > rejected_scores).sum()
        ranking_accuracy = float(correct_ranking) / batch_size
        
        # Calculate the average margin
        margin = chosen_scores - rejected_scores
        avg_margin = float(margin.mean())
        
        # Calculate the distribution of each pipeline being chosen as preferred
        pipeline_preferred = {
            "basic_preferred": float((chosen_idx == 0).sum()) / batch_size,
            "intermediate_preferred": float((chosen_idx == 1).sum()) / batch_size,
            "advanced_preferred": float((chosen_idx == 2).sum()) / batch_size
        }
        
        # Calculate the distribution of each pipeline being rejected
        pipeline_rejected = {
            "basic_rejected": float((rejected_idx == 0).sum()) / batch_size,
            "intermediate_rejected": float((rejected_idx == 1).sum()) / batch_size,
            "advanced_rejected": float((rejected_idx == 2).sum()) / batch_size
        }
        
        return {
            "ranking_accuracy": ranking_accuracy,  # Main metric: correct ranking ratio
            "avg_margin": avg_margin,  # Average margin
            **pipeline_preferred,  # Distribution of preferred pipelines
            **pipeline_rejected,  # Distribution of rejected pipelines
        }
        
    def train(self):
        try:
            self.load_model_and_tokenizer()
            tokenized_datasets = self.prepare_dataset()
            
            # Calculate the appropriate number of evaluation steps
            num_train_samples = len(tokenized_datasets["train"])
            # batch_size = 8
            num_gpu = torch.cuda.device_count()
            steps_per_epoch = num_train_samples // (8 * num_gpu)
            
            # Set the checkpoint and training log save path
            checkpoint_dir = self.save_dir / "qwen_pairwise_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = checkpoint_dir / "training.log"
            
            # Record the training start information
            self._log_to_file(
                f"Training started with:\n"
                f"- Number of GPUs: {num_gpu}\n"
                f"- Number of training samples: {num_train_samples}\n"
                f"- Number of validation samples: {len(tokenized_datasets['validation'])}\n"
                # f"- Steps per epoch: {steps_per_epoch}\n"
            )
            
            # Configure training parameters
            training_args = TrainingArguments(
                output_dir=str(checkpoint_dir),
                # Training strategy configuration
                per_device_train_batch_size=self.pairwise_config["per_device_train_batch_size"],
                per_device_eval_batch_size=self.pairwise_config["per_device_eval_batch_size"],
                gradient_accumulation_steps=self.pairwise_config["gradient_accumulation_steps"],
                num_train_epochs=self.pairwise_config["num_train_epochs"],
                learning_rate=self.pairwise_config["learning_rate"],
                lr_scheduler_type=self.pairwise_config["lr_scheduler_type"],
                fp16=self.pairwise_config["fp16"],
                
                # Evaluation strategy configuration
                eval_strategy=self.pairwise_config["eval_strategy"],
                eval_steps=self.pairwise_config.get("eval_steps", None),  # If "epoch" mode then no need
                
                # Save strategy configuration
                save_strategy=self.pairwise_config["save_strategy"],
                save_steps=self.pairwise_config.get("save_steps", None),  # If "epoch" mode then no need
                
                # Log related configuration
                logging_dir=str(self.config.logs_dir / "pairwise" / "qwen_pairwise"),
                logging_strategy=self.pairwise_config["logging_strategy"],
                logging_steps=self.pairwise_config.get("logging_steps", None),  # If "epoch" mode then no need
                logging_first_step=self.pairwise_config["logging_first_step"],
                report_to=["tensorboard"],
                
                # Checkpoint related configuration
                load_best_model_at_end=self.pairwise_config["load_best_model_at_end"],
                metric_for_best_model="ranking_accuracy",
                greater_is_better=self.pairwise_config["greater_is_better"],
                save_total_limit=self.pairwise_config["save_total_limit"],
                
                # Other configuration
                ddp_find_unused_parameters=self.pairwise_config["ddp_find_unused_parameters"],
                local_rank=self.local_rank,
                dataloader_num_workers=self.pairwise_config["dataloader_num_workers"],
                remove_unused_columns=self.pairwise_config["remove_unused_columns"],
                warmup_ratio=self.pairwise_config["warmup_ratio"],
                weight_decay=self.pairwise_config["weight_decay"],
                max_grad_norm=self.pairwise_config["max_grad_norm"]
            )
            
            # Create a trainer with logging functionality
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                compute_metrics=self.compute_metrics,
                callbacks=[TrainingCallback(self._log_to_file)]  # Add callback to record training process
            )
            
            # Start training
            train_result = trainer.train()
            
            # Final evaluation
            final_metrics = trainer.evaluate()
            
            # Record the final evaluation results
            self._log_to_file("\n\n\nFinal Evaluation Results:")
            for metric_name, value in final_metrics.items():
                self._log_to_file(f"{metric_name}: {value:.4f}")
            
            # Save the final model and results
            if self.local_rank == 0:
                # Save the final model
                final_model_dir = self.save_dir / "final_model_classifier"
                trainer.save_model(final_model_dir)
                
                # Save the classifier head configuration and weights
                classifier_config = self.model.get_classifier_config()
                classifier_state = self.model.classifier.state_dict()
                
                classifier_save = {
                    "config": classifier_config,
                    "state_dict": classifier_state
                }
                
                torch.save(classifier_save, final_model_dir / "classifier.pt")
                
                # Save the training results
                results_file = checkpoint_dir / "training_results.json"
                results_data = {
                    "train_results": train_result.metrics,
                    "eval_results": final_metrics,
                    "train_samples": num_train_samples,
                    "eval_samples": len(tokenized_datasets["validation"]),
                    "training_args": training_args.to_dict(),
                    "classifier_config": classifier_config  # Also included in the training results
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
    """The callback for recording the training process"""
    def __init__(self, log_func):
        self.log_func = log_func
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.log_func(f"\n\n<< Epoch {state.epoch:.2f} started >>")
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Extract metrics
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
        
        # Build the evaluation log
        eval_log = (
            f"Evaluation at Step {state.global_step} (epoch {state.epoch:.2f}):\n"
            # f"[Raw Accuracy      ] {raw_accuracy:.5f}\n"
            # f"[Penalty          ] {penalty:.5f}\n"
            # f"[Penalized Acc    ] {penalized_accuracy:.5f}\n"
            # f"[Acc for Classes  ] "
            # f"Basic: {class_accuracies[0]:.5f} | "
            # f"Inter: {class_accuracies[1]:.5f} | "
            # f"Advan: {class_accuracies[2]:.5f}\n"
            # f"[Predict Label Dist] "
            # f"Basic: {pred_dist[0]:.5f} | Inter: {pred_dist[1]:.5f} | Advan: {pred_dist[2]:.5f}\n"
            # f"[Actual Label Dist ] "
            # f"Basic: {true_dist[0]:.5f} | Inter: {true_dist[1]:.5f} | Advan: {true_dist[2]:.5f}"
        )
        
        self.log_func(eval_log)
        
    def on_log(self, args, state, control, logs, **kwargs):
        # Record training loss and other information
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