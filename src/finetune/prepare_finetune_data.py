import json
from pathlib import Path
from typing import Dict, List
import random
from ..core.config import Config

class FinetuneDataProcessor:
    """处理标注数据生成微调数据集"""
    
    def __init__(self):
        self.config = Config()
        # 创建finetune数据目录
        self.finetune_dir = self.config.data_dir / "finetune"
        self.finetune_dir.mkdir(parents=True, exist_ok=True)
        
    def load_labeled_data(self, labeled_file: str) -> List[Dict]:
        """加载标注数据"""
        data = []
        with open(labeled_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
        
    def create_prompt(self, item: Dict) -> str:
        """构造输入prompt"""
        # 将schema格式化为字符串
        schema_str = self._format_schema(item["enhanced_linked_schema_wo_info"])
        
        # 构造prompt模板
        prompt = f"""Based on the following question and database schema, classify which SQL generation pipeline should be used (1: VANILLA, 2: DC_REFINER, 3: DC_OS_REFINER).

Question: {item["question"]}

Database Schema:
{schema_str}

Classification:"""
        
        return prompt
        
    def _format_schema(self, schema: Dict) -> str:
        """格式化schema为字符串"""
        result = []
        for table in schema["tables"]:
            # 添加表名
            result.append(f"Table: {table['table']}")
            # 添加列
            result.append(f"Columns: {', '.join(table['columns'])}")
            # 添加主键(如果存在)
            if "primary_keys" in table:
                result.append(f"Primary keys: {', '.join(table['primary_keys'])}")
            result.append("")  # 空行分隔
            
        return "\n".join(result)
        
    def prepare_data(self, labeled_file: str, train_ratio: float = 0.8):
        """准备训练和验证数据集"""
        # 加载标注数据
        data = self.load_labeled_data(labeled_file)
        
        # 构造训练样本
        samples = []
        for item in data:
            prompt = self.create_prompt(item)
            label = item["label"]  # 使用数字标签
            
            samples.append({
                "prompt": prompt,
                "response": str(label),  # 转换为字符串
                "question_id": item["question_id"]
            })
            
        # 随机打乱数据
        random.shuffle(samples)
        
        # 划分训练集和验证集
        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        valid_samples = samples[split_idx:]
        
        # 保存数据集
        train_file = self.finetune_dir / "train.json"
        valid_file = self.finetune_dir / "valid.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
            
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
            
        print(f"Total samples: {len(samples)}")
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(valid_samples)}")
        print(f"Data saved to {self.finetune_dir}")

def main():
    processor = FinetuneDataProcessor()
    labeled_file = "data/labeled/sampled_bird_demo_pipeline_preference.jsonl"
    processor.prepare_data(labeled_file)

if __name__ == "__main__":
    main() 