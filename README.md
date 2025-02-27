<h1 align="center">🐘ElephantSQL: Task-Aware Routing towards <br>Cost-Efficient Natural Language to SQL Generation</h1>

Natural Language to SQL (NL2SQL) systems allow non-technical users to query databases seamlessly. Despite the success of advanced LLM-based NL2SQL approaches on leaderboards, their substantial computational costs—often overlooked—stand as the "elephant in the room" in current leaderboard-driven research, making them economically impractical for real-world deployment and widespread adoption. To tackle this, we present ElephantSQL, a task-aware routing framework that assigns queries to suitable SQL generation methods based on task complexity. We design and investigate five routers, including SFT-tuned lightweight language models as classifiers and DPO for preference learning, to direct simple queries to efficient approaches while reserving computationally intensive methods for complex cases. Drawing from economics, we introduce the Token Elasticity of Performance (TEP), a metric capturing cost-efficiency by quantifying how sensitively NL2SQL performance responds to token investment. Our experiments on the BIRD dataset show that with the Qwen2.5-0.5B-DPO router, ElephantSQL matches the performance of the most advanced methods in our study while cutting token use by over 40%, achieving more than a 2× boost in TEP over non-routing approaches. This not only advances the pursuit of cost-efficient NL2SQL but also invites the community to weigh resource efficiency alongside performance, fostering sustainable progress in this field.

## 📂Project Structure

```
🐘ElephantSQL/
├── config/                     # Configuration files
│   ├── config_example.yaml     # A provided config example
│   ├── config.yaml             # Config file for LLM API keys and needed paths
│   ├── sft_config.yaml         # Config for Qwen2.5-0.5b LoRA SFT
│   └── roberta_config.yaml     # Config for RoBERTa-base full SFT
├── data/                       # Data files and databases
├── examples/                   # Examples for usage
├── scripts/                    # Scripts
├── src/                        # Source code for the project
│   ├── core/                   # Core functionalities and utilities
│   ├── evaluation/             # Evaluation scripts for various processes
│   ├── modules/                # Modular components for NL2SQL
│   │   ├── post_processing/    # Post-processing modules
│   │   ├── schema_linking/     # Schema linking modules
│   │   ├── sql_generation/     # SQL generation modules
│   │   └── base.py             # Base classes for modules
│   ├── router/                 # Implemnetation of Routers
│   ├── sft/                   	# Data preparation for SFT and SFT codes
│   ├── pipeline.py             # Pipeline management for processing queries
│   ├── pipeline_factory.py     # Factory for creating different pipeline of levels
│   └── run.py                  # Runner for the 🐘ElephantSQL system
└── README.md                   # This Readme
```

