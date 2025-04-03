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

def load_sft_config(config_name: str):
    """Load the SFT configuration"""
    config_path = Path("config") / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class RoBERTaClassifierTrainer:
    def __init__(self, sft_dataset: str, sft_config: str, training_mode: str = 'lora'):
        self.config = Config()
        self.model_path = self.config.roberta_dir
        self.sft_dataset_dir = self.config.sft_data_dir / sft_dataset
        self.save_dir = self.config.roberta_save_dir
        self.training_mode = training_mode
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.log_file = None
        
        # Load the specified configuration file
        self.sft_config = load_sft_config(sft_config)
        if self.local_rank == 0:
            print(f"Using SFT config: {sft_config}")
            print(f"Training mode: {training_mode}")
            
    def _log_to_file(self, message: str):
        """Write the log to the file"""
        if self.log_file and self.local_rank == 0:  # Only write the log in the main process
            with open(self.log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
                
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.model_path,
            padding_side="right"
        )
        
        # Load the classification model
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=3,  # Basic, Intermediate, Advanced
            problem_type="single_label_classification"
        )
        
        if self.training_mode == 'lora':
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,  # Use the sequence classification task type
                r=self.sft_config["lora_r"],
                lora_alpha=self.sft_config["lora_alpha"],
                lora_dropout=self.sft_config["lora_dropout"],
                target_modules=self.sft_config["target_modules"],
                bias="none",
            )
            
            # Prepare the model
            self.model = get_peft_model(self.model, lora_config)
            
            if self.local_rank == 0:
                print("Using LoRA fine-tuning")
                # Print the parameter training status
                print("\nParameter training status:")
                print("Classifier parameters:")
                for name, param in self.model.classifier.named_parameters():
                    print(f"  {name}: requires_grad = {param.requires_grad}")
        else:
            if self.local_rank == 0:
                print("Using full parameter fine-tuning")
        
    def prepare_dataset(self):
        """Prepare the dataset"""
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
            # Ensure the labels are of the correct type and range
            labels = [int(label) for label in examples["label"]]
            # Verify the label range and print the abnormal values
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
        """Compute the evaluation metrics, adding a penalty for incorrectly classifying complex questions into simpler pipelines"""
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = predictions.argmax(-1)  # Convert to 0-based labels
        labels = eval_pred.label_ids  # 0-based labels
        
        total = len(labels)
        correct = 0
        penalty = 0
        penalty_factor = self.sft_config["penalty_factor"]  # Penalty factor
        
        # Calculate the statistics for each class
        class_correct = {i: 0 for i in range(3)}
        class_total = {i: 0 for i in range(3)}
        pred_dist = {i: 0 for i in range(3)}
        
        for pred, label in zip(predictions, labels):
            class_total[label] += 1
            pred_dist[pred] += 1
            
            if pred == label:
                correct += 1
                class_correct[label] += 1
            elif label > pred and pred == 0:
                # Penalize the case of classifying Intermediate and Advanced into Basic
                penalty += 1 * penalty_factor
        
        # Calculate the penalized accuracy
        penalized_accuracy = (correct - penalty) / total
        
        # Calculate the accuracy for each class
        class_accuracy = {}
        for i in range(3):
            if class_total[i] > 0:
                class_accuracy[f"class_{i}_accuracy"] = float(class_correct[i]) / class_total[i]
            else:
                class_accuracy[f"class_{i}_accuracy"] = 0.0
        
        # Calculate the percentage of distribution
        pred_dist_pct = {
            f"pred_dist_{i}": float(count) / total 
            for i, count in pred_dist.items()
        }
        true_dist_pct = {
            f"true_dist_{i}": float(class_total[i]) / total 
            for i in range(3)
        }
        
        # Record the penalty information
        penalty_info = {
            "penalty": float(penalty),
            "raw_accuracy": float(correct) / total,
            "penalized_accuracy": float(penalized_accuracy)
        }
        
        # Return all metrics
        return {
            "accuracy": penalized_accuracy,  # Use the penalized accuracy
            **class_accuracy,
            **pred_dist_pct,
            **true_dist_pct,
            **penalty_info
        }
        
    def train(self):
        try:
            self.load_model_and_tokenizer()
            tokenized_datasets = self.prepare_dataset()
            
            # Set the checkpoint and training log save path
            checkpoint_dir = self.save_dir / f"roberta_classifier_checkpoints_{self.training_mode}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = checkpoint_dir / "training.log"
            
            # Record the training start information
            self._log_to_file(
                f"Training started with:\n"
                f"- Number of training samples: {len(tokenized_datasets['train'])}\n"
                f"- Number of validation samples: {len(tokenized_datasets['validation'])}\n"
            )
            
            # Configure the training parameters
            training_args = TrainingArguments(
                output_dir=str(checkpoint_dir),
                per_device_train_batch_size=self.sft_config["per_device_train_batch_size"],
                per_device_eval_batch_size=self.sft_config["per_device_eval_batch_size"],
                gradient_accumulation_steps=self.sft_config["gradient_accumulation_steps"],
                num_train_epochs=self.sft_config["num_train_epochs"],
                learning_rate=self.sft_config["learning_rate"],
                lr_scheduler_type=self.sft_config["lr_scheduler_type"],
                fp16=self.sft_config["fp16"] if self.training_mode == 'lora' else False,
                
                # Evaluation strategy configuration
                eval_strategy=self.sft_config["eval_strategy"],
                eval_steps=self.sft_config.get("eval_steps", None),
                
                # Save strategy configuration
                save_strategy=self.sft_config["save_strategy"],
                save_steps=self.sft_config.get("save_steps", None),
                
                # Log related configuration
                logging_dir=str(self.config.logs_dir / "sft" / f"roberta_classifier_{self.training_mode}"),
                logging_strategy=self.sft_config["logging_strategy"],
                logging_steps=self.sft_config.get("logging_steps", None),
                logging_first_step=self.sft_config["logging_first_step"],
                report_to=["tensorboard"],
                
                # Checkpoint related configuration
                load_best_model_at_end=self.sft_config["load_best_model_at_end"],
                metric_for_best_model=self.sft_config["metric_for_best_model"],
                greater_is_better=self.sft_config["greater_is_better"],
                save_total_limit=self.sft_config["save_total_limit"],
                
                # Other configuration
                ddp_find_unused_parameters=self.sft_config["ddp_find_unused_parameters"],
                local_rank=self.local_rank,
                dataloader_num_workers=self.sft_config["dataloader_num_workers"],
                remove_unused_columns=self.sft_config["remove_unused_columns"],
                warmup_ratio=self.sft_config["warmup_ratio"],
                weight_decay=self.sft_config["weight_decay"],
                max_grad_norm=self.sft_config["max_grad_norm"]
            )
            
            # Create the trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                compute_metrics=self.compute_metrics,
                callbacks=[TrainingCallback(self._log_to_file)]
            )
            
            # Start training
            train_result = trainer.train()
            
            # Final evaluation
            final_metrics = trainer.evaluate()
            
            # Record the final evaluation results
            self._log_to_file("\nFinal Evaluation Results:")
            for metric_name, value in final_metrics.items():
                self._log_to_file(f"{metric_name}: {value:.4f}")
            
            # Save the final model and results
            if self.local_rank == 0:
                # Save the final model
                final_model_dir = self.save_dir / f"final_model_roberta_{self.training_mode}"
                trainer.save_model(final_model_dir)
                
                # Save the classifier head configuration and weights
                classifier_config = {
                    "num_labels": 3,
                    "hidden_size": self.model.config.hidden_size,
                    "classifier_dropout": self.model.config.hidden_dropout_prob,
                    "model_type": "RobertaForSequenceClassification"
                }
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
                    "train_samples": len(tokenized_datasets["train"]),
                    "eval_samples": len(tokenized_datasets["validation"]),
                    "training_args": training_args.to_dict()
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
        # Record the training loss and other information
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
    parser.add_argument('--training_mode', type=str, default='lora',
                       help='Training mode: lora or full')
    args = parser.parse_args()
    
    is_distributed = False
    try:
        is_distributed = setup_distributed()
        trainer = RoBERTaClassifierTrainer(
            sft_config=args.sft_config,
            sft_dataset=args.sft_dataset,
            training_mode=args.training_mode
        )
        trainer.train()
    finally:
        if is_distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main() 