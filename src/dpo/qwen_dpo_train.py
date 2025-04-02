import os
import argparse
import yaml
from pathlib import Path
# -----------------------------
# 环境变量设置：只使用 GPU4,5,6,7（内部设备编号会重排为 0,1,2,3）
# 同时设置 NCCL 环境变量以解决 RTX 4000 系列显卡的通信问题
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

import copy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset
from trl.trainer.dpo_config import DPOConfig
from ..core.config import Config
from trl import DPOTrainer
from transformers.trainer_callback import TrainerCallback


# =============================
# 自定义回调：将日志保存到 txt 文件
# =============================
class SaveLogCallback(TrainerCallback):
    def __init__(self, log_file="training_log.txt"):
        self.log_file = log_file
        # 初始化时清空文件，并写入表头
        with open(self.log_file, "w") as f:
            f.write("step, train_loss, eval_loss\n")
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            # 获取训练 loss 和 eval_loss，如果没有则为 NA
            train_loss = logs.get("loss", "NA")
            eval_loss = logs.get("eval_loss", "NA")
            # 将日志追加到文件中
            with open(self.log_file, "a") as f:
                f.write(f"{step}, {train_loss}, {eval_loss}\n")

# =============================
# 自定义 Trainer 类：针对分类任务的 DPOTrainer
# =============================
class ClassificationDPOTrainer(DPOTrainer):
    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        """
        对 "prompt" 进行分词，生成键 "prompt_input_ids"；
        同时，将原始的整数标签保存为单元素列表，分别存入 "chosen_input_ids" 和 "rejected_input_ids"。
        这里的标签取值为 0, 1, 2。
        """
        # 对 prompt 进行分词，设定最大长度和 padding
        prompt_encoded = processing_class(
            features["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_prompt_length,
            add_special_tokens=add_special_tokens,
        )
        prompt_input_ids = prompt_encoded["input_ids"]
        # 将整数标签包装为单元素列表（后续 collator 会将其转换为张量形状 [batch, 1]）
        chosen_input_ids = [features["chosen"]]
        rejected_input_ids = [features["rejected"]]
        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    def concatenated_forward(self, model, batch):
        """
        前向传播步骤：
          1. 从 batch 中取出分词后的 prompt_input_ids 及 attention_mask（如果没有，则自动构造）。
          2. 调用模型进行前向传播，得到 logits（形状 [batch, num_labels]）。
          3. 计算 log_softmax 得到各类别的对数概率。
          4. 根据 batch 中保存的标签（保存在 chosen_input_ids 和 rejected_input_ids，均为 [batch, 1]）利用 gather
             获取对应的 log-probability。
          5. 为满足 TRL 内部统计要求，还返回 "mean_chosen_logits" 和 "mean_rejected_logits"（这里直接设为对应 log-prob）。
        """
        input_ids = batch["prompt_input_ids"]  # [batch, seq_len]
        # 如果 DataCollator 已经生成了 attention mask，则使用之；否则自动生成
        attention_mask = batch.get("prompt_attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != self.args.padding_value).long()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, num_labels]
        logprobs = torch.log_softmax(logits, dim=-1)  # [batch, num_labels]
        # 将单元素列表转换为张量（形状 [batch, 1]）后 squeeze 得到 [batch] 的标签
        chosen_label = batch["chosen_input_ids"].squeeze(1)
        rejected_label = batch["rejected_input_ids"].squeeze(1)
        # 利用 gather 获取每个样本对应标签的 log-probability
        chosen_logps = logprobs.gather(1, chosen_label.unsqueeze(1)).squeeze(1)
        rejected_logps = logprobs.gather(1, rejected_label.unsqueeze(1)).squeeze(1)
        # 返回字典中增加 mean_chosen_logits 和 mean_rejected_logits，供后续统计和 loss 计算使用
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": chosen_logps,  # 分类任务下，用 logps 代替
            "mean_rejected_logits": rejected_logps,
        }
    
def load_dpo_config(config_name: str):
    """加载DPO配置"""
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
    # 1. 模型和 Tokenizer 的加载及 pad_token 设置
    # =============================
    
    model_name = config.qwen_dir
    # 加载模型配置，指定 num_labels=3（类别：0,1,2），模型将自动添加分类头
    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 如果 tokenizer 没有 pad_token，则使用 eos_token；若都没有则添加新 token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
    # 更新 pad_token_id
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 构造参考模型（ref_model），用于计算对比损失。复制后冻结参数。
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # =============================
    # 2. 数据预处理
    # =============================
    # 假设数据文件中每个样本格式为：
    #   { "text": "……", "chosen": 0, "rejected": 1 }
    # 其中 "chosen" 与 "rejected" 为整数标签（类别），取值范围 0,1,2。
    # dataset = load_dataset(
    #     'json',
    #     data_files={
    #         'train': "data/dpo/bird_dev_dataset/classifier_train.json",
    #         'validation': "data/dpo/bird_dev_dataset/classifier_valid.json"
    #     }
    # )

    dataset = load_dataset("json", data_files=dpo_config.train_file)["train"]
    e_dataset = load_dataset("json", data_files=dpo_config.valid_file)["train"]

    def preprocess(example):
        return {
            "prompt": str(example.get("text", "")),
            "chosen": int(example.get("chosen", 0)),
            "rejected": int(example.get("rejected", 0)),
        }

    # 使用 dataset.map 逐条处理数据，同时删除原有所有列，只保留预处理后的键。
    # tokenized_datasets = dataset.map(
    #     preprocess,
    #     batched=True,
    #     remove_columns=dataset["train"].column_names,
    #     num_proc=None
    # )

    dataset = dataset.map(preprocess, batched=False, remove_columns=dataset.column_names)
    e_dataset = e_dataset.map(preprocess, batched=False, remove_columns=e_dataset.column_names)

    # =============================
    # 3. 定义训练参数（DPOConfig）
    # =============================
    dpo_train_config = DPOConfig(
        output_dir=dpo_config.output_dir,  # 模型和检查点保存目录    
        per_device_train_batch_size=dpo_config.per_device_train_batch_size,    # 每个 GPU 的 batch size（根据显存调整）
        num_train_epochs=dpo_config.num_train_epochs,                          # 训练轮数
        fp16=dpo_config.fp16,                                   # 启用 FP16 混合精度训练
        learning_rate=dpo_config.learning_rate,                          # 学习率
        loss_type=dpo_config.loss_type,                         # DPO 损失类型，此处使用 sigmoid loss
        beta=dpo_config.beta,                                    # 控制偏离参考模型的程度
        max_prompt_length=dpo_config.max_prompt_length,                       # prompt 的最大长度
        max_completion_length=dpo_config.max_completion_length,                   # 本任务中不使用，但必须给定
        max_length=dpo_config.max_length,                              # prompt 的最大总长度（用于分词）
        padding_value=dpo_config.padding_value,        # 填充 token 的 id
        generate_during_eval=dpo_config.generate_during_eval,                  # 分类任务无需生成
        dataset_num_proc=dpo_config.dataset_num_proc,                       # 禁用多进程，确保逐条处理
        remove_unused_columns=dpo_config.remove_unused_columns,                 # 保留所有数据列，避免 Trainer 自动移除自定义标签列
        logging_steps=dpo_config.logging_steps,                            # 每 50 步打印一次日志
        save_steps=dpo_config.save_steps,                              # 每 100 步保存一次模型
        # save_total_limit=5,
        # metric_for_best_model='loss',
        # greater_is_better=False,
        eval_strategy=dpo_config.eval_strategy,  # 每隔一定步数评估一次
        eval_steps=dpo_config.eval_steps,           # 每 100 步进行评估（与保存间隔相同）
        logging_dir=dpo_config.logging_dir,  # 日志保存目录
    )

    # =============================
    # 4. 初始化 Trainer 并开始训练
    # =============================
    trainer = ClassificationDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_train_config,
        train_dataset=dataset,
        eval_dataset=e_dataset,
        processing_class=tokenizer,  # 使用原始 tokenizer，内部 tokenize_row 会处理标签
        data_collator=None,          # 默认使用 DataCollatorForPreference
    )

    # # 添加自定义回调，将日志写入 txt 文件
    # trainer.add_callback(SaveLogCallback(log_file="training_log.txt"))

    # 添加自定义 TXT 日志回调，日志将保存到指定文件
    trainer.add_callback(SaveLogCallback(log_file=os.path.join(dpo_config.output_dir, "training_log.txt")))

    # 开始训练
    trainer.train()

    final_metrics = trainer.evaluate()

    # 训练结束后，模型和检查点保存至 dpo_config.output_dir 指定的目录
    print(f"模型和检查点保存至: {dpo_config.output_dir}")

