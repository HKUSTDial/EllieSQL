
# bash scripts/routing.sh

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m src.run

# wait
# python -m src.evaluation.compute_EX