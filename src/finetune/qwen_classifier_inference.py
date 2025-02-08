import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from ..core.config import Config
from .qwen_classifier_sft import QwenForSequenceClassification
from .instruction_templates import PipelineClassificationTemplates

def set_seed(seed: int = 42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 禁用cudnn的随机性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class QwenClassifier:
    """用于分类推理的Qwen分类器: """

    def __init__(self, seed: int = 42):
        # 设置随机种子
        set_seed(seed)
        
        self.config = Config()
        self.model_path = self.config.model_dir
        self.lora_path = self.config.finetune_save_dir / "final_model_classifier"
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
        
        # 加载基础模型
        base_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 创建分类模型并设置为float16
        self.model = QwenForSequenceClassification(base_model)
        self.model.classifier = self.model.classifier.half()
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(
            self.model,
            self.lora_path,
            torch_dtype=torch.float16
        )
        
        # 设置为评估模式
        self.model.eval()
        
        # 禁用dropout等随机行为
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
        
    def classify(self, question: str, schema: dict) -> tuple[int, dict]:
        """进行分类预测，返回预测的类别和概率分布"""
        # 使用模板创建输入文本
        input_text = self.templates.create_classifier_prompt(question, schema)
        
        # 编码输入
        inputs = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 将输入移动到正确的设备和数据类型
        inputs = {k: v.to(self.model.device, dtype=torch.long) for k, v in inputs.items()}
        
        # 进行预测
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 使用softmax获取概率分布
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            
            # 获取每个类别的概率
            probabilities = {
                "basic": float(probs[0][0]),
                "intermediate": float(probs[0][1]),
                "advanced": float(probs[0][2])
            }
            
        # 转换回1-based标签
        return predicted_class + 1, probabilities

def test_deterministic():
    """测试分类结果的确定性"""
    classifier = QwenClassifier(seed=42)
    
    # 测试样例
    question = "How many pets have a greater weight than 10?"
    schema = {
        "tables": [{
            "table": "Pets",
            "columns": ["PetID", "weight"],
            "primary_keys": ["PetID"]
        }]
    }
    
    # 多次运行分类，确保结果一致
    results = []
    for _ in range(5):
        label, probabilities = classifier.classify(question, schema)
        results.append((label, probabilities))
        print(f"Iteration {_+1}: {label}, Probabilities: {probabilities}")
    
    # 验证所有结果是否相同
    assert len(set(label for label, _ in results)) == 1, "分类结果不确定！"
    print("确定性测试通过：所有结果一致")

if __name__ == "__main__":
    test_deterministic() 