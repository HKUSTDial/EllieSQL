<h1 align="center">🐘ElephantSQL: Task-Aware Routing for Cost-Efficient NL2SQL Generation</h1>

Natural Language to SQL (NL2SQL) is to convert natural language queries into structured SQL queries, enabling non-technical users to interact with databases effortlessly. This capability holds immense significance as it democratizes data access, empowering individuals and organizations to leverage data-driven insights without requiring demanding specialized technical expertise. However, in the field of Natural Language to SQL (NL2SQL), the high operational costs of existing high-performing solutions remain the ***"elephant in the room"***, posing a significant barrier to their practicality and widespread deployment. To mitigate this critical issue, we propose ElephantSQL, aiming to bridge the gap between NL2SQL technology and real-world applications by introducing a cost-efficient framework. ElephantSQL employs a multi-pipeline architecture that intelligently routes tasks to specialized SQL generation modules based on their complexity. Simpler tasks are processed through lightweight pipelines, while more complex queries are handled by more advanced modules. By strategically allocating resources in this way, ElephantSQL not only enhances scalability and efficiency but also significantly reduces operational costs, making NL2SQL more accessible and practical for real-world use.

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

