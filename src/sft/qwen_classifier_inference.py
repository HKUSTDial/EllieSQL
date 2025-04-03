import json
from pathlib import Path
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
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable the randomness of cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class QwenClassifier:
    """Qwen classifier for classification inference"""

    def __init__(self, lora_path: str = None, seed: int = 42):
        # Set the random seed
        set_seed(seed)
        
        self.config = Config()
        self.model_path = self.config.qwen_dir
        self.lora_path = Path(lora_path) if lora_path else self.config.sft_save_dir / "final_model_classifier"
        self.templates = PipelineClassificationTemplates()
        
        # Load the model and tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # NOTE: Load the classifier head from the manually saved classifier.pt, 
        #       ensure the same classifier head is used for training and inference
        # Load the classifier head configuration and weights
        classifier_path = self.lora_path / "classifier.pt"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Cannot find the classifier file: {classifier_path}")
        
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
        
        # Disable the random behavior of dropout, etc.
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
        
    def classify(self, question: str, schema: dict) -> tuple[int, dict]:
        """Perform classification prediction, return the predicted class and probability distribution"""
        # Use the template to create the input text
        input_text = self.templates.create_classifier_prompt(question, schema)
        # print(input_text)
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
            self.model.eval()
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Use softmax to get the probability distribution
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

def test():
    """Test the classifier inference"""
    classifier = QwenClassifier(
        lora_path="/data/zhuyizhang/saves/Qwen2.5-0.5B-router/sft/final_model_classifier",
        seed=42
    )
    
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

    # Classify each test case
    print(f"Supposed to be: 1, 1, 1, 3, 3")
    for i, test_case in enumerate(test_cases, 1):
        label, probabilities = classifier.classify(test_case["question"], test_case["schema"])
        print(f"[Test Case {i}] Predicted Label: {label}, Probabilities: {probabilities}")
    

if __name__ == "__main__":
    test() 