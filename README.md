<h1 align="center">🐘EllieSQL: Cost-Efficient Text-to-SQL <br> with Complexity-Aware Routing</h1>

<div align="center">

Official repository for the paper *"EllieSQL: Cost-Efficient Text-to-SQL with Complexity-Aware Routing"*.

[![Homepage](https://img.shields.io/badge/🏠-Homepage-blue)](https://elliesql.github.io/)
[![COLM 2025](https://img.shields.io/badge/COLM-2025-green.svg)](https://colmweb.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.22402-b31b1b.svg)](https://arxiv.org/abs/2503.22402)
[![Checkpoints](https://img.shields.io/badge/🤗-Checkpoints-orange)](https://huggingface.co/derrickzhu/EllieSQL_Router_Checkpoints)
[![Python](https://img.shields.io/badge/Python-3.12.7-3776AB.svg?style=flat)](https://www.python.org/downloads/release/python-3127/)


</div>

![teaser](asserts/teaser.png)


## 📢 News
[Jul. 08, 2025] 🎉EllieSQL is accepted by COLM 2025!

[Apr. 03, 2025] 🐘EllieSQL is publicly released!

## 📖 Overview

Text-to-SQL automatically translates natural language queries to SQL, allowing non-technical users to retrieve data from databases without specialized SQL knowledge. Despite the success of advanced LLM-based Text-to-SQL approaches on leaderboards, their unsustainable computational costs—often overlooked—stand as the "elephant in the room" in current leaderboard-driven research, limiting their economic practicability for real-world deployment and widespread adoption. 

To tackle this, we exploratively propose EllieSQL, a complexity-aware routing framework that assigns queries to suitable SQL generation pipelines based on estimated complexity. We investigate multiple routers to direct simple queries to efficient approaches while reserving computationally intensive methods for complex cases. Drawing from economics, we introduce the Token Elasticity of Performance (TEP) metric, capturing cost-efficiency by quantifying the responsiveness of performance gains relative to token investment in SQL generation. Experiments show that compared to always using the most advanced methods in our study, EllieSQL with the Qwen2.5-0.5B-DPO router reduces token use by over 40% without compromising performance on Bird development set, achieving more than a 2× boost in TEP over non-routing approaches. This not only advances the pursuit of cost-efficient Text-to-SQL but also invites the community to weigh resource efficiency alongside performance, contributing to progress in sustainable Text-to-SQL.


## 📂 Project Structure

```
🐘EllieSQL/
├── config/                     # Configuration files
│   ├── config_example.yaml     # A provided config example
│   ├── config.yaml             # Config file for LLM API keys and needed paths
│   └── xxx_config.yaml         # Config for SFT
├── data/                       # Data files and databases
├── examples/                   # Examples for usage
├── scripts/                    # Bash scripts
│   ├── exp/                    # Scripts for experiments: run base/routing experiments
│   ├── sft/                    # Scripts for router SFT
│   └── eval/                   # Scripts for evaluation (e.g., compute EX)
├── src/                        # Source code for the project
│   ├── core/                   # Core functionalities and utilities
│   ├── evaluation/             # Evaluation scripts
│   ├── modules/                # Modular components for Text-to-SQL
│   │   ├── schema_linking/     # Schema linking modules
│   │   ├── sql_generation/     # SQL generation modules
│   │   ├── post_processing/    # Post-processing modules
│   │   └── base.py             # Base classes for modules
│   ├── router/                 # Implementation of various routers
│   ├── sft/                    # Data preparation for SFT and implementation of SFT
│   ├── dpo/                    # Data preparation for DPO and implementation of DPO
│   ├── pipeline.py             # Pipeline management for processing queries
│   ├── pipeline_factory.py     # Factory for creating different pipeline of levels
│   ├── run_base.py             # Runner for the base methods: G_B, G_M, G_A
│   └── run_routing.py          # Runner for the 🐘EllieSQL with routers
└── README.md                   # This Readme
```

## ⚙ Environment Setup

1. Create and activate a conda environment:

   ```bash
   conda create --name elliesql python=3.12.7
   conda activate elliesql
   cd EllieSQL
   pip install -r requirements.txt
   ```

2. Download required resources:

   - Bird dataset: [Bird Official Website](https://bird-bench.github.io/)
   - Router checkpoints: Available on [🤗Hugging Face](https://huggingface.co/derrickzhu/EllieSQL_Router_Checkpoints)

3. Pre-process dataset file into the format required for this project:

   ```bash
   python -m data.preprocess
   ```

4. Configure your settings:

   ```bash
   cp config/config_example.yaml config/config.yaml
   ```

   Edit `config/config.yaml` to set up your:

   - API Base Url
   - API keys
   - Dataset paths
   - Model paths
   - Other configurations

   ```yaml
   # Inference with proprietary LLMs via API
   api:
     base_url: "your_base_url"                                # e.g., "https://api.openai.com/v1"
     api_key: "your_api_key"                                  # e.g., "sk-xxxxxxxxxxxxxxxxxxxxxx"
   
   # Inference with local open-source LLMs is also supported
   local_models:
     "qwen2.5-coder-7b-instruct":
       path: "/path/to/Qwen2.5-Coder-7B-Instruct"
       prompt_format: "qwen"
     "xxxxx":                                                 # you can also add new models
       .....
   
   # Set paths and directories
   paths:
     data_dir: "./data"                                       # root for data file
     database_dir: "./data/databases"                         # dir to databases in Bird
     results_dir: "./results"                                 # dir to store results
     logs_dir: "./logs"                                       # dir to store logs
     qwen_dir: "/path/to/Qwen2.5-0.5B"                        # dir to base model to train routers
     roberta_dir: "/path/to/roberta-base"
     sft_data_dir: "./data/sft"                               # dir to prepared data to fine-tune routers
     pairwise_data_dir: "./data/pairwise"
     cascade_data_dir: "./data/cascade"
     dpo_data_dir: "./data/dpo"
     sft_save_dir: "/path/to/saves/Qwen2.5-0.5B-router/sft"   # dir to saved weights after fine-tuning
     roberta_save_dir: "/path/to/saves/RoBERTa-router"
     pairwise_save_dir: "/path/to/saves/Qwen2.5-0.5B-router/pairwise"
     cascade_qwen_save_dir: "/path/to/saves/Qwen2.5-0.5B-router/cascade"
     cascade_roberta_save_dir: "/path/to/saves/RoBERTa-router/cascade"
     dpo_save_dir: "/path/to/saves/Qwen2.5-0.5B-router/dpo"
   ```

## 💻 Hardware Requirements

- Training: 4 × NVIDIA RTX 4090
- Inference: 1 × NVIDIA RTX 4090

## 🧪 Running Experiments

### Base Methods

1. Ensure your configuration in `config/config.yaml` is correct
2. Run base experiments:

```bash
bash scripts/exp/run_base.sh
```

### Routing Methods

1. Download router checkpoints
2. Update checkpoint paths in `scripts/exp/run_routing.sh`
3. Run routing experiments:

```bash
bash scripts/exp/run_routing.sh
```

Note: Remember to specify the correct GPU in the scripts according to your device.

## ⚗️ Training Routers

We provide scripts for training different types of routers:

- Classification-based routers
- Cascading routers
- Preference learning-based routers

Example: Training a RoBERTa-based classifier router:

```bash
bash scripts/sft/roberta_classifier_sft.sh
```

You can also change hyperparameters in fine-tuning by editing SFT configs in  `config/`.

Note: Remember to specify the correct GPU in the scripts according to your device.

## 📝 Citation

If you find our work useful or inspiring, please kindly cite:

```
@misc{zhu2025elliesql,
      title={EllieSQL: Cost-Efficient Text-to-SQL with Complexity-Aware Routing}, 
      author={Yizhang Zhu and Runzhi Jiang and Boyan Li and Nan Tang and Yuyu Luo},
      year={2025},
      eprint={2503.22402},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2503.22402}, 
}
```
