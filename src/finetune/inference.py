import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from ..core.config import Config

class QwenClassifier:
    def __init__(self):
        self.config = Config()
        self.model_path = self.config.model_dir
        self.lora_path = self.config.finetune_save_dir / "final_model"
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.lora_path,
            torch_dtype=torch.float16
        )
        
    def classify(self, question: str, schema: dict) -> int:
        """进行分类预测"""
        # 构造prompt
        prompt = self._create_prompt(question, schema)
        
        # 生成回答
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                # temperature=0.1, # 贪婪解码无需设置temperature
                do_sample=False,  # 使用贪婪解码
                num_beams=1,      # 使用贪婪搜索
                pad_token_id=self.tokenizer.eos_token_id,  # 设置pad_token_id
                eos_token_id=self.tokenizer.eos_token_id,  # 设置eos_token_id
                num_return_sequences=1
            )
            
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # 转换为标签
        try:
            label = int(response)
            if label not in [1, 2, 3]:
                label = 3  # 默认使用最复杂的pipeline
        except:
            label = 3
            
        return label
        
    def _create_prompt(self, question: str, schema: dict) -> str:
        """创建分类prompt"""
        # 格式化schema
        schema_str = []
        for table in schema["tables"]:
            schema_str.append(f"Table: {table['table']}")
            schema_str.append(f"Columns: {', '.join(table['columns'])}")
            if "primary_keys" in table:
                schema_str.append(f"Primary keys: {', '.join(table['primary_keys'])}")
            schema_str.append("")
            
        schema_str = "\n".join(schema_str)
        
        # 构造prompt
        prompt = f"""Based on the following question and database schema, classify which SQL generation pipeline should be used (1: VANILLA, 2: DC_REFINER, 3: DC_OS_REFINER).

Question: {question}

Database Schema:
{schema_str}

Classification:"""
        
        return prompt

# 使用示例
def main():
    # 初始化分类器
    classifier = QwenClassifier()
    
    # 测试样例
    question = "How many pets have a greater weight than 10?"
    schema = {
        "tables": [{
            "table": "Pets",
            "columns": ["PetID", "weight"],
            "primary_keys": ["PetID"]
        }]
    }
    
    # 进行分类
    label = classifier.classify(question, schema)
    print(f"Question: {question}")
    print(f"Predicted pipeline: {label}")

if __name__ == "__main__":
    main() 