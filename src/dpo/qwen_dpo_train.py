import os
import argparse
import yaml
from pathlib import Path
import copy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset
from trl.trainer.dpo_config import DPOConfig
from ..core.config import Config
from trl import DPOTrainer
from transformers.trainer_callback import TrainerCallback


# =============================
# Custom callback: Save logs to txt file
# =============================
class SaveLogCallback(TrainerCallback):
    def __init__(self, log_file="training_log.txt"):
        self.log_file = log_file
        # Clear the file when initialized, and write the header
        with open(self.log_file, "w") as f:
            f.write("step, train_loss, eval_loss\n")
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            # Get the training loss and eval_loss, if not, set to NA
            train_loss = logs.get("loss", "NA")
            eval_loss = logs.get("eval_loss", "NA")
            # Append the log to the file
            with open(self.log_file, "a") as f:
                f.write(f"{step}, {train_loss}, {eval_loss}\n")

# =============================
# Custom Trainer class: For classification tasks
# =============================
class ClassificationDPOTrainer(DPOTrainer):
    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        """
        Tokenize the "prompt", generate the key "prompt_input_ids";
        At the same time, save the original integer labels as single-element lists, and store them in "chosen_input_ids" and "rejected_input_ids".
        The labels here are 0, 1, 2.
        """
        # Tokenize the prompt, set the maximum length and padding
        prompt_encoded = processing_class(
            features["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_prompt_length,
            add_special_tokens=add_special_tokens,
        )
        prompt_input_ids = prompt_encoded["input_ids"]
        # Wrap the integer labels as single-element lists (the collator will convert them to tensor shape [batch, 1])
        chosen_input_ids = [features["chosen"]]
        rejected_input_ids = [features["rejected"]]
        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    def concatenated_forward(self, model, batch):
        """
        Forward propagation steps:
          1. Get the tokenized prompt_input_ids and attention_mask (if not, construct automatically) from the batch.
          2. Call the model for forward propagation, get the logits (shape [batch, num_labels]).
          3. Calculate the log_softmax to get the log-probability of each category.
          4. Use gather to get the corresponding log-probability based on the labels saved in the batch (saved in chosen_input_ids and rejected_input_ids, both are [batch, 1]).
          5. Return "mean_chosen_logits" and "mean_rejected_logits" (set directly to the corresponding log-prob) to meet the internal statistics requirements of TRL.
        """
        input_ids = batch["prompt_input_ids"]  # [batch, seq_len]
        # If the DataCollator has already generated the attention mask, use it; otherwise, generate it automatically
        attention_mask = batch.get("prompt_attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != self.args.padding_value).long()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, num_labels]
        logprobs = torch.log_softmax(logits, dim=-1)  # [batch, num_labels]
        # Convert the single-element lists to tensors (shape [batch, 1]) and squeeze to get the labels [batch]
        chosen_label = batch["chosen_input_ids"].squeeze(1)
        rejected_label = batch["rejected_input_ids"].squeeze(1)
        # Use gather to get the log-probability of each sample corresponding to the label
        chosen_logps = logprobs.gather(1, chosen_label.unsqueeze(1)).squeeze(1)
        rejected_logps = logprobs.gather(1, rejected_label.unsqueeze(1)).squeeze(1)
        # Add "mean_chosen_logits" and "mean_rejected_logits" to the dictionary for subsequent statistics and loss calculation
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": chosen_logps,  # In the classification task, use logps instead of logits
            "mean_rejected_logits": rejected_logps,
        }
    
def load_dpo_config(config_name: str):
    """Load DPO configuration"""
    config_path = Path("config") / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo_config', type=str, default='dpo_config',
                       help='Name of the DPO config file under config/ (without .yaml)')

    args = parser.parse_args()
    
    config = Config()
    dpo_config = load_dpo_config(args.dpo_config)

        
    # =============================
    # 1. Load the model and Tokenizer and set the pad_token
    # =============================
    
    model_name = config.qwen_dir
    # Load the model configuration, specify num_labels=3 (categories: 0,1,2), the model will automatically add a classification head
    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # If the tokenizer does not have a pad_token, use the eos_token; if neither, add a new token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
    # Update the pad_token_id
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Construct the reference model (ref_model), for calculating the contrastive loss. Copy and freeze the parameters.
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # =============================
    # 2. Data preprocessing
    # =============================
    # Assume the format of each sample in the data file is:
    #   { "text": "……", "chosen": 0, "rejected": 1 }
    #   where "chosen" and "rejected" are integer labels (categories), with values 0,1,2.
    # dataset = load_dataset(
    #     'json',
    #     data_files={
    #         'train': "data/dpo/bird_dev_dataset/classifier_train.json",
    #         'validation': "data/dpo/bird_dev_dataset/classifier_valid.json"
    #     }
    # )

    dataset = load_dataset("json", data_files=dpo_config["train_file"])["train"]
    e_dataset = load_dataset("json", data_files=dpo_config["valid_file"])["train"]

    def preprocess(example):
        return {
            "prompt": str(example.get("text", "")),
            "chosen": int(example.get("chosen", 0)),
            "rejected": int(example.get("rejected", 0)),
        }

    # Use dataset.map to process the data line by line, while deleting all existing columns and only keeping the processed keys.
    # tokenized_datasets = dataset.map(
    #     preprocess,
    #     batched=True,
    #     remove_columns=dataset["train"].column_names,
    #     num_proc=None
    # )

    dataset = dataset.map(preprocess, batched=False, remove_columns=dataset.column_names)
    e_dataset = e_dataset.map(preprocess, batched=False, remove_columns=e_dataset.column_names)

    # =============================
    # 3. Define training parameters (DPOConfig)
    # =============================
    dpo_train_config = DPOConfig(
        output_dir=dpo_config["output_dir"],  # The directory to save the model and checkpoints
        per_device_train_batch_size=dpo_config["per_device_train_batch_size"],    # The batch size for each GPU (adjust according to the memory)
        num_train_epochs=dpo_config["num_train_epochs"],                          # The number of training epochs
        fp16=dpo_config["fp16"],                                   # Enable FP16 mixed precision training
        learning_rate=dpo_config["learning_rate"],                          # The learning rate
        loss_type=dpo_config["loss_type"],                         # The loss type of DPO, here using sigmoid loss
        beta=dpo_config["beta"],                                    # The degree of deviation from the reference model
        max_prompt_length=dpo_config["max_prompt_length"],                       # The maximum length of the prompt
        max_completion_length=dpo_config["max_completion_length"],                   # This task is not used, but must be given
        max_length=dpo_config["max_length"],                              # The maximum total length of the prompt (for tokenization)
        padding_value=tokenizer.pad_token_id, #dpo_config["padding_value"],        # The id of the padding token
        generate_during_eval=dpo_config["generate_during_eval"],                  # The classification task does not need to generate
        dataset_num_proc=None,#dpo_config["dataset_num_proc"],                       # Disable multi-processing
        remove_unused_columns=dpo_config["remove_unused_columns"],                 # Keep all data columns, avoid Trainer automatically removing custom label columns
        logging_steps=dpo_config["logging_steps"],                            # Print logs every 50 steps
        save_steps=dpo_config["save_steps"],                              # Save the model every 100 steps
        # save_total_limit=5,
        # metric_for_best_model='loss',
        # greater_is_better=False,
        eval_strategy=dpo_config["eval_strategy"],  # Evaluate every certain steps
        eval_steps=dpo_config["eval_steps"],           # Evaluate every 100 steps (same as the save interval)
        logging_dir=dpo_config["logging_dir"],  # The directory to save the logs
    )

    # =============================
    # 4. Initialize Trainer and start training
    # =============================
    trainer = ClassificationDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_train_config,
        train_dataset=dataset,
        eval_dataset=e_dataset,
        processing_class=tokenizer,  # Use the original tokenizer, the internal tokenize_row will handle the labels
        data_collator=None,          # Default using DataCollatorForPreference
    )

    # # Add a custom callback to write logs to a txt file
    # trainer.add_callback(SaveLogCallback(log_file="training_log.txt"))

    # Add a custom TXT log callback, the logs will be saved to the specified file
    trainer.add_callback(SaveLogCallback(log_file=os.path.join(dpo_config["output_dir"], "training_log.txt")))

    # Start training
    trainer.train()

    final_metrics = trainer.evaluate()

    outdir = dpo_config["output_dir"]

    # After training, the model and checkpoints will be saved to the directory specified in dpo_config.output_dir
    print(f"The model and checkpoints will be saved to: {outdir}")

