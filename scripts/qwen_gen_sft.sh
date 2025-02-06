
# bash scripts/qwen_gen_sft.sh

# 运行数据处理脚本
python -m src.finetune.prepare_gen_sft_data

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 在多卡RTX 4090上分布式训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=29500 -m src.finetune.qwen_gen_sft

# 推理示例
python -m src.finetune.qwen_gen_inference