import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..core.utils import load_json, load_jsonl
from ..sft.instruction_templates import PipelineClassificationTemplates
from pathlib import Path



class DPOClassifierRouter(RouterBase):
    """Qwen router with classification head trained by DPO"""
    
    def __init__(self, name: str = "DPOClassifierRouter", seed: int = 42, model_path: str = None):
        super().__init__(name)
        self.config = Config()
        self.templates = PipelineClassificationTemplates()

        self.model_path = Path(model_path) if model_path else self.config.dpo_save_dir / "final_model_classifier"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.trained_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # Set the random seed to ensure reproducibility
        self._set_seed(seed)

        # Set the GPU device, prioritize CUDA
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # If there are multiple GPUs, use DataParallel to wrap the model, use gpu0,1,2,3
        if torch.cuda.device_count() > 1:
            self.trained_model = torch.nn.DataParallel(self.trained_model, device_ids=[0, 1, 2, 3])
        self.trained_model.to(self.device)
    

    def _set_seed(self, seed: int):
        """Set the random seed"""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def _predict(self, question: str, schema: dict) -> tuple[int, dict]:
        """Use dpo-qwen model to predict"""
        input_text = self.templates.create_classifier_prompt(
                    question=question,
                    schema=schema
                )
        
        self.trained_model.eval()  # Switch to evaluation mode

        # Use tokenizer to tokenize the prompt
        encoded_input = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        # Model output logits
        with torch.no_grad():
            outputs = self.trained_model(**encoded_input)
            logits = outputs.logits
            # Use softmax to calculate the predicted probability
            probs = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()

        # print(f"Test prompt: {input_text}")
        # print(f"Predicted label: {predicted_label}, Probabilities: {probs.squeeze().tolist()}")

        print(f"Predicted label: {predicted_label}, Probabilities: {probs.squeeze().tolist()}")
        ans = predicted_label
        return ans

        # if(ans == "BASIC"):
        #     return 0
        # elif(ans == "INTERMEDIATE"):
        #     return 1
        # else:
        #     return 2
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """Select the appropriate SQL generator based on the prediction result of the classifier"""
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # Use classifier to predict
        predicted_class = self._predict(query, linked_schema)

        
        # Select the pipeline based on the prediction result
        if predicted_class == 0:  # Basic
            return PipelineLevel.BASIC.value
        elif predicted_class == 1:  # Intermediate
            return PipelineLevel.INTERMEDIATE.value
        elif predicted_class == 2:  # Advanced
            return PipelineLevel.ADVANCED.value
        else:
            raise ValueError(f"Invalid prediction from classifier: {predicted_class}") 