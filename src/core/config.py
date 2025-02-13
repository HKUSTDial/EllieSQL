import yaml
from pathlib import Path

class Config:
    """配置管理类"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 加载配置文件
            config_path = Path("config/config.yaml")
            with open(config_path, "r") as f:
                cls._instance.config = yaml.safe_load(f)
        return cls._instance
    
    @property
    def data_dir(self) -> Path:
        return Path(self.config["paths"]["data_dir"])
        
    @property
    def database_dir(self) -> Path:
        return Path(self.config["paths"]["database_dir"])
        
    @property
    def results_dir(self) -> Path:
        return Path(self.config["paths"]["results_dir"])
    
    @property
    def intermediate_results_dir(self) -> Path:
        return self.results_dir / "intermediate_results"
        
    @property
    def logs_dir(self) -> Path:
        return Path(self.config["paths"]["logs_dir"])
        
    @property 
    def gold_schema_linking_dir(self) -> Path:
        return Path(self.config["paths"]["gold_schema_linking_dir"]) 
    
    @property
    def qwen_dir(self) -> Path:
        return Path(self.config["paths"]["qwen_dir"])
    
    @property
    def roberta_dir(self) -> Path:
        return Path(self.config["paths"]["roberta_dir"])
    
    @property
    def sft_data_dir(self) -> Path:
        return Path(self.config["paths"]["sft_data_dir"])
    
    @property
    def sft_save_dir(self) -> Path:
        return Path(self.config["paths"]["sft_save_dir"])
    
    @property
    def dpo_data_dir(self) -> Path:
        return Path(self.config["paths"]["dpo_data_dir"])
    
    @property
    def dpo_save_dir(self) -> Path:
        return Path(self.config["paths"]["dpo_save_dir"])