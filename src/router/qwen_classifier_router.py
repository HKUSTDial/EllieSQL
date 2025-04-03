from typing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..sft.qwen_classifier_sft import QwenForSequenceClassification
from ..sft.instruction_templates import PipelineClassificationTemplates
from pathlib import Path

class QwenClassifierRouter(RouterBase):
    """Qwen router with classification head trained by LoRA SFT"""
    
    def __init__(self, name: str = "QwenClassifierRouter", lora_path: str = None, seed: int = 42):
        super().__init__(name)
        self.config = Config()
        self.model_path = self.config.qwen_dir
        # Use the specified LoRA path or the default path
        self.lora_path = Path(lora_path) if lora_path else self.config.sft_save_dir / "final_model_classifier"
        self.templates = PipelineClassificationTemplates()
        
        # Set the random seed to ensure reproducibility
        self._set_seed(seed)
        
        # Load the model and tokenizer
        self._load_model_and_tokenizer()
        
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
        
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the classifier head configuration and weights
        classifier_path = self.lora_path / "classifier.pt"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Cannot find the classifier head file: {classifier_path}")
        
        classifier_save = torch.load(classifier_path, map_location='cpu', weights_only=True)
        classifier_config = classifier_save["config"]
        classifier_state = classifier_save["state_dict"]
        
        # Load the base model
        base_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Create the classification model using the saved configuration
        self.model = QwenForSequenceClassification(
            base_model,
            num_labels=classifier_config["num_labels"]
        )
        
        # Load the classifier head weights
        self.model.classifier.load_state_dict(classifier_state)
        
        # Convert to float16
        self.model.classifier = self.model.classifier.half()
        
        # Load the LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model,
            self.lora_path,
            torch_dtype=torch.float16
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        # Disable dropout and other random behaviors
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
                
    def _predict(self, question: str, schema: dict) -> tuple[int, dict]:
        """Use the classification head model to predict"""
        input_text = self.templates.create_classifier_prompt(question, schema)
        
        inputs = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.model.device, dtype=torch.long) for k, v in inputs.items()}
        
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get the probability distribution using softmax
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            
            # Get the probability of each class
            probabilities = {
                "basic": float(probs[0][0]),
                "intermediate": float(probs[0][1]),
                "advanced": float(probs[0][2])
            }
            
        return predicted_class, probabilities
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """Select the appropriate SQL generator based on the prediction result of the classifier"""
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # Use the classifier to predict
        predicted_class, probabilities = self._predict(query, linked_schema)
        
        # Record the prediction probabilities
        self.logger.info(f"Pipeline selection probabilities for query {query_id}:")
        for pipeline, prob in probabilities.items():
            self.logger.info(f"  {pipeline}: {prob:.4f}")
        
        # Select the pipeline based on the prediction result
        if predicted_class == 0:  # Basic
            return PipelineLevel.BASIC.value
        elif predicted_class == 1:  # Intermediate
            return PipelineLevel.INTERMEDIATE.value
        elif predicted_class == 2:  # Advanced
            return PipelineLevel.ADVANCED.value
        else:
            raise ValueError(f"Invalid prediction from classifier: {predicted_class}") 