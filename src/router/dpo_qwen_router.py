from typing import Dict
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config

from ..core.utils import load_json, load_jsonl
from ..sft.instruction_templates import PipelineClassificationTemplates
# 如果使用非simplified，应该使用sft的template

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification




class DPOClassifierRouter(RouterBase):
    """添加了分类头并使用DPO训练的Qwen路由器"""
    
    def __init__(self, name: str = "DPOClassifierRouter", seed: int = 42):
        super().__init__(name)
        self.config = Config()
        self.templates = PipelineClassificationTemplates()

        #self.model_name = "data/dpo/bird_dev_dataset/dpo-qwen2.5-0.5b_second/checkpoint-378"
        #self.model_name = "/data/jiangrunzhi/saves/Qwen2.5-0.5B-router/dpo/bird_train_dataset/dpo-qwen2.5-0.5b/checkpoint-1800"
        #self.model_name = "/data/jiangrunzhi/saves/Qwen2.5-0.5B-router/dpo/bird_train_dataset/dpo-qwen2.5-0.5b_second/checkpoint-2211"
        #self.model_name = "/data/jiangrunzhi/saves/Qwen2.5-0.5B-router/dpo/bird_train_dataset/dpo-qwen2.5-0.5b_second/checkpoint-2200"
        # self.model_name = "/data/jiangrunzhi/saves/Qwen2.5-0.5B-router/dpo/bird_train_dataset/dpo-qwen2.5-0.5b_0219/checkpoint-3600"
        self.model_name = "/data/jiangrunzhi/saves/Qwen2.5-0.5B-router/dpo/bird_train_dataset_simplified/dpo-qwen2.5-0.5b_0220/checkpoint-3300"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.trained_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)



    def _predict(self, question: str, schema: dict) -> tuple[int, dict]:
        """使用dpo-qwen模型进行预测"""
        input_text = self.templates.create_classifier_prompt(
                    question=question,
                    schema=schema
                )
        
        self.trained_model.eval()  # 切换到评估模式

        # 使用 tokenizer 对 prompt 分词
        encoded_input = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        # 模型输出 logits
        with torch.no_grad():
            outputs = self.trained_model(**encoded_input)
            logits = outputs.logits
            # 使用 softmax 计算预测概率
            probs = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()

        # print(f"测试 prompt: {input_text}")
        # print(f"预测类别: {predicted_label}，各类别概率: {probs.squeeze().tolist()}")

        print(f"预测类别: {predicted_label}，各类别概率: {probs.squeeze().tolist()}")
        ans = predicted_label
        return ans

        # if(ans == "BASIC"):
        #     return 0
        # elif(ans == "INTERMEDIATE"):
        #     return 1
        # else:
        #     return 2
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """根据分类器预测结果选择合适的SQL生成器"""
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # 使用分类器进行预测
        predicted_class = self._predict(query, linked_schema)

        
        # 根据预测结果选择pipeline
        if predicted_class == 0:  # Basic
            return PipelineLevel.BASIC.value
        elif predicted_class == 1:  # Intermediate
            return PipelineLevel.INTERMEDIATE.value
        elif predicted_class == 2:  # Advanced
            return PipelineLevel.ADVANCED.value
        else:
            raise ValueError(f"Invalid prediction from classifier: {predicted_class}") 