from typing import Dict
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..sft.instruction_templates import PipelineClassificationTemplates


class RoBERTaClassifierRouter(RouterBase):
    """Router using RoBERTa for classification"""
    
    def __init__(self, name: str = "RoBERTaClassifierRouter", model_path: str = None, seed: int = 42):
        super().__init__(name)
        self.config = Config()
        # Use the specified model path or the default path
        self.model_path = Path(model_path) if model_path else self.config.roberta_save_dir / "final_model_roberta"
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
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}")
            
        # Load the tokenizer from the original pre-trained model
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.config.roberta_dir,  # Use the tokenizer of the pre-trained model
            padding_side="right"
        )
        
        # Load the fine-tuned classification model
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=3,  # Basic, Intermediate, Advanced
            problem_type="single_label_classification"
        )
        
        # Move the model to GPU and set to evaluation mode
        self.model = self.model.cuda()
        self.model.eval()
        
        # Disable dropout and other random behaviors
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
                
    def _predict(self, question: str, schema: dict) -> tuple[int, dict]:
        """Use the RoBERTa model for prediction"""
        # Use the template to create the input text
        input_text = self.templates.create_classifier_prompt(question, schema)
        
        # Encode the input
        inputs = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move the input to GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Perform prediction
        with torch.no_grad():
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
            
        # Convert back to 1-based labels
        return predicted_class + 1, probabilities
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """Select the appropriate SQL generator based on the prediction result of the RoBERTa classifier"""
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # Use the classifier for prediction
        predicted_class, probabilities = self._predict(query, linked_schema)
        
        # Record the prediction probabilities
        self.logger.info(f"Pipeline selection probabilities for query {query_id}:")
        for pipeline, prob in probabilities.items():
            self.logger.info(f"  {pipeline}: {prob:.4f}")
        
        # Select the pipeline based on the prediction result
        if predicted_class == 1:  # Basic
            return PipelineLevel.BASIC.value
        elif predicted_class == 2:  # Intermediate
            return PipelineLevel.INTERMEDIATE.value
        elif predicted_class == 3:  # Advanced
            return PipelineLevel.ADVANCED.value
        else:
            raise ValueError(f"Invalid prediction from classifier: {predicted_class}") 