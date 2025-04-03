import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from ..core.config import Config
from .qwen_cascade_sft import QwenForBinaryClassification
from .instruction_templates import PipelineClassificationTemplates

def set_seed(seed: int = 42):
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable the randomness of cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class QwenCascadeClassifier:
    """Qwen classifier for binary classification inference"""

    def __init__(self, pipeline_type: str, model_name: str, confidence_threshold: float = 0.5, seed: int = 42):
        # 设置随机种子
        set_seed(seed)
        
        self.config = Config()
        self.pipeline_type = pipeline_type
        self.model_path = self.config.cascade_qwen_save_dir / f"final_model_{model_name}"
        self.templates = PipelineClassificationTemplates()
        
        # Add the confidence threshold parameter
        self.confidence_threshold = confidence_threshold
        
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
        
        # Load the classifier configuration and weights
        classifier_path = self.model_path / "classifier.pt"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Cannot find the classifier file: {classifier_path}")
        
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
        
        # Disable the random behavior of dropout, etc.
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
        
        # Move the input to the correct device and data type
        inputs = {k: v.to(self.model.device, dtype=torch.long) for k, v in inputs.items()}
        
        # Perform prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Use softmax to get the probability distribution
            probs = F.softmax(logits, dim=-1)
            # Get the probability of the positive class (can be processed)
            positive_prob = float(probs[0][1])
            
            # Only when the positive class probability exceeds the threshold is it considered to be processed
            can_handle = (positive_prob >= self.confidence_threshold)
            
        return can_handle, positive_prob

def test(pipeline_type: str, confidence_threshold: float):
    """Test the binary classifier inference"""
    # Test cases
    test_cases = [
        {
            "question": "How many pets have a greater weight than 10?",
            "schema": {
                "tables": [{
                    "table": "Pets",
                    "columns": ["PetID", "weight"],
                    "primary_keys": ["PetID"]
                }]
            }
        },
        {
            "question": "Give the publisher ID of Star Trek.",
            "schema": {
                "tables": [{
                    "table": "publisher",
                    "columns": ["publisher_name", "id"],
                    "primary_keys": ["id"]
                }]
            }
        },
        {
            "question": "What is the border color of card \"Ancestor's Chosen\"?",
            "schema": {
                "tables": [{
                    "table": "cards",
                    "columns": ["id", "borderColor", "name"],
                    "primary_keys": ["id"]
                }]
            }
        },
        {
            "question": "How many patients with an Ig G higher than normal?",
            "schema": {
                "tables": [{
                    "table": "Laboratory",
                    "columns": ["ID", "IGG", "Date"],
                    "primary_keys": ["ID", "Date"]
                }, {
                    "table": "Patient",
                    "columns": ["ID"],
                    "primary_keys": ["ID"]
                }]
            }
        },
        {
            "question": "What is the eligible free or reduced price meal rate for the top 5 schools in grades 1-12 with the highest free or reduced price meal count of the schools with the ownership code 66?",
            "schema": {
                "tables": [{
                    "table": "frpm",
                    "columns": ["Enrollment (K-12)", "CDSCode", "FRPM Count (K-12)"],
                    "primary_keys": ["CDSCode"]
                }, {
                    "table": "schools",
                    "columns": ["SOC", "CDSCode"],
                    "primary_keys": ["CDSCode"]
                }]
            }
        }
    ]

    # Select the pipeline type to test
    model_name = f"{pipeline_type}_classifier"
    
    classifier = QwenCascadeClassifier(
        pipeline_type=pipeline_type,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        seed=42
    )

    # Classify each test case
    print(f"\nTesting {pipeline_type.upper()} Pipeline Classifier (confidence threshold: {confidence_threshold}):")
    print(f"Expected: True for basic questions, False for others" if pipeline_type == "basic" else
          f"Expected: True for intermediate questions, False for advanced/unsolved" if pipeline_type == "intermediate" else
          f"Expected: True for advanced questions, False for unsolved")
    
    for i, test_case in enumerate(test_cases, 1):
        can_handle, confidence = classifier.classify(test_case["question"], test_case["schema"])
        print(f"[Test Case {i}] Can handle by {pipeline_type} pipeline: {can_handle} "
              f"(confidence: {confidence:.3f}, threshold: {confidence_threshold})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_type", type=str, default="basic", 
                       choices=["basic", "intermediate", "advanced"])
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Confidence threshold for positive predictions")
    args = parser.parse_args()
    
    test(args.pipeline_type, args.confidence_threshold) 