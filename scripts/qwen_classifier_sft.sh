
# bash scripts/qwen_classifier_sft.sh

# 运行数据处理脚本
python -m src.finetune.prepare_classifier_sft_data

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 在多卡RTX 4090上分布式训练
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=29500 -m src.finetune.qwen_classifier_sft

# 推理示例
# python -m src.finetune.qwen_classifier_inference

# tensorboard可视化
# tensorboard --logdir logs/sft/qwen_classifier