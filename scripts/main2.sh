# bash scripts/main2.sh

# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 在多卡RTX 4090上分布式训练
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "-------------------------------- Run examples.main --------------------------------"
python -m examples.main_local

# wait
# echo "-------------------------------- Compute EX stats --------------------------------"
# python -m src.evaluation.compute_EX

