import torch
import torch.nn.functional as F
from typing import Dict
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..sft.instruction_templates import PipelineClassificationTemplates

class RoBERTaCascadeClassifier:
    """Cascade binary classifier based on RoBERTa"""
    
    def __init__(self, pipeline_type: str, model_name: str, model_path: Path, confidence_threshold: float = 0.5):
        self.config = Config()
        self.pipeline_type = pipeline_type
        self.model_path = model_path / f"final_model_{model_name}"
        self.confidence_threshold = confidence_threshold
        self.templates = PipelineClassificationTemplates()
        
        # Load the model and tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Cannot find the model file: {self.model_path}")
            
        # Load the tokenizer from the original pre-trained model
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.config.roberta_dir,
            padding_side="right"
        )
        
        # Load the base model
        base_model = RobertaForSequenceClassification.from_pretrained(
            self.config.roberta_dir,
            num_labels=2,  # Binary classification
            problem_type="single_label_classification"
        )
        
        # Load the LoRA weights
        self.model = PeftModel.from_pretrained(
            base_model,
            self.model_path,
            torch_dtype=torch.float16
        )
        
        # Move the model to GPU and set to evaluation mode
        self.model = self.model.cuda()
        self.model.eval()
        
        # Disable dropout and other random behaviors
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
                
    def classify(self, question: str, schema: dict) -> tuple[bool, float]:
        """Perform binary classification prediction, return whether it can be processed and confidence"""
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
            # Get the probability of the positive class (can be processed)
            positive_prob = float(probs[0][1])
            
            # Only consider it can be processed when the positive class probability exceeds the threshold
            can_handle = (positive_prob >= self.confidence_threshold)
            
        return can_handle, positive_prob

class RoBERTaCascadeRouter(RouterBase):
    """Cascade RoBERTa router: judge in the order of basic->intermediate->advanced"""
    
    def __init__(
        self,
        name: str = "RoBERTaCascadeRouter",
        confidence_threshold: float = 0.5,
        model_path: str = None,
        seed: int = 42
    ):
        super().__init__(name)
        self.config = Config()
        self.model_path = Path(model_path) if model_path else self.config.cascade_roberta_save_dir
        
        # Set the random seed
        self._set_seed(seed)
        
        # Initialize the classifier for the corresponding pipeline
        self.basic_classifier = RoBERTaCascadeClassifier(
            pipeline_type="basic",
            model_name="basic_classifier",
            model_path=self.model_path,
            confidence_threshold=confidence_threshold
        )
        print("[i] Successfully load and initialize classifier for basic pipeline")
        
        self.intermediate_classifier = RoBERTaCascadeClassifier(
            pipeline_type="intermediate",
            model_name="intermediate_classifier",
            model_path=self.model_path,
            confidence_threshold=confidence_threshold
        )
        print("[i] Successfully load and initialize classifier for intermediate pipeline")

        # intermediate cannot be processed, directly use advanced, no additional judgment is needed
        # self.advanced_classifier = RoBERTaCascadeClassifier(
        #     pipeline_type="advanced",
        #     model_name="advanced_classifier",
        #     model_path=self.model_path,
        #     confidence_threshold=confidence_threshold
        # )
        # print("Successfully load and initialize classifier for advanced pipeline")
        
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
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """Cascade routing logic"""
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # 1. First try the basic pipeline
        can_handle, confidence = self.basic_classifier.classify(query, linked_schema)
        if can_handle:
            self.logger.info(f"Query {query_id} routed to BASIC pipeline (confidence: {confidence:.3f})")
            return PipelineLevel.BASIC.value
            
        # 2. If basic fails, try intermediate pipeline
        can_handle, confidence = self.intermediate_classifier.classify(query, linked_schema)
        if can_handle:
            self.logger.info(f"Query {query_id} routed to INTERMEDIATE pipeline (confidence: {confidence:.3f})")
            return PipelineLevel.INTERMEDIATE.value
            
        # 3. If intermediate fails, directly use advanced pipeline
        # Optional: You can also check if advanced pipeline can handle it first
        # can_handle, confidence = self.advanced_classifier.classify(query, linked_schema)
        self.logger.info(
            f"Query {query_id} routed to ADVANCED pipeline (confidence: {confidence:.3f})"
            # f"({'can handle' if can_handle else 'cannot handle'}, confidence: {confidence:.3f})"
        )
        return PipelineLevel.ADVANCED.value
        