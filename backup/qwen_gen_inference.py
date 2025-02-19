from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from ..core.config import Config
from .instruction_templates import PipelineClassificationTemplates

class QwenGenerator:
    def __init__(self, lora_path: str = None):
        self.config = Config()
        self.model_path = self.config.qwen_dir
        self.lora_path = Path(lora_path) if lora_path else self.config.sft_save_dir / "final_model_gen"
        self.templates = PipelineClassificationTemplates()
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.lora_path,
            torch_dtype=torch.float16
        )
        
    def classify(self, question: str, schema: dict) -> int:
        """进行分类预测"""
        # 使用模板创建输入文本
        prompt = self.templates.create_generator_prompt(question, schema)
        
        # 生成回答
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
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

# 使用示例
def main():
    # 初始化分类器
    classifier = QwenGenerator()
    
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