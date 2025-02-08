# 从BASIC标签(1)中采样5个样本
python -m src.router.sample_labeled_data data/labeled/bird_dev_pipeline_label.jsonl --label 1 --sample_size 5

# 从INTERMEDIATE标签(2)中采样3个样本
python -m src.router.sample_labeled_data data/labeled/bird_dev_pipeline_label.jsonl --label 2 --sample_size 3

# 从ADVANCED标签(3)中采样10个样本，使用不同的随机种子
python -m src.router.sample_labeled_data data/labeled/bird_dev_pipeline_label.jsonl --label 3 --sample_size 10 --seed 123

# 查看UNSOLVED标签(4)的所有样本
python -m src.router.sample_labeled_data data/labeled/bird_dev_pipeline_label.jsonl --label 4 --sample_size 999