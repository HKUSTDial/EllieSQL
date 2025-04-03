import torch
import torch.nn.functional as F
from typing import Dict
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..sft.qwen_cascade_sft import QwenForBinaryClassification
from ..sft.instruction_templates import PipelineClassificationTemplates

class QwenCascadeClassifier:
    """Cascade binary classifier based on Qwen"""
    
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
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.qwen_dir,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the classifier head configuration and weights
        classifier_path = self.model_path / "classifier.pt"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Cannot find the classifier head file: {classifier_path}")
        
        classifier_save = torch.load(classifier_path, map_location='cpu', weights_only=True)
        classifier_config = classifier_save["config"]
        classifier_state = classifier_save["state_dict"]
        
        # Load the base model
        base_model = AutoModel.from_pretrained(
            self.config.qwen_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Create the classifier model using the saved configuration
        self.model = QwenForBinaryClassification(base_model)
        
        # Load the classifier head weights
        self.model.classifier.load_state_dict(classifier_state)
        
        # Convert to float16
        self.model.classifier = self.model.classifier.half()
        
        # Load the LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model,
            self.model_path,
            torch_dtype=torch.float16
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        # Disable dropout and other random behaviors
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
                
    def classify(self, question: str, schema: dict) -> tuple[bool, float]:
        """Perform binary classification prediction, return whether it can be processed and confidence"""
        # Create the input text using the template
        input_text = self.templates.create_classifier_prompt(question, schema)
        
        # Encode the input
        inputs = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move the input to the correct device and data type
        inputs = {k: v.to(self.model.device, dtype=torch.long) for k, v in inputs.items()}
        
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

class QwenCascadeRouter(RouterBase):
    """Cascade Qwen router: judge in the order of basic->intermediate->advanced"""
    
    def __init__(
        self,
        name: str = "QwenCascadeRouter",
        confidence_threshold: float = 0.5,
        model_path: str = None,
        seed: int = 42
    ):
        super().__init__(name)
        self.config = Config()
        self.model_path = Path(model_path) if model_path else self.config.cascade_qwen_save_dir
        
        # Set the random seed
        self._set_seed(seed)
        
        # Initialize the classifier for the corresponding pipeline
        self.basic_classifier = QwenCascadeClassifier(
            pipeline_type="basic",
            model_name="basic_classifier",
            model_path=self.model_path,
            confidence_threshold=confidence_threshold
        )
        print("[i] Successfully load and initialize classifier for basic pipeline")
        
        self.intermediate_classifier = QwenCascadeClassifier(
            pipeline_type="intermediate",
            model_name="intermediate_classifier",
            model_path=self.model_path,
            confidence_threshold=confidence_threshold
        )
        print("[i] Successfully load and initialize classifier for intermediate pipeline")
        
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
        self.logger.info(f"Query {query_id} routed to ADVANCED pipeline (confidence: {confidence:.3f})")
        return PipelineLevel.ADVANCED.value 