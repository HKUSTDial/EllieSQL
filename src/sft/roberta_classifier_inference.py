import json
from pathlib import Path
import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from ..core.config import Config
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

class RoBERTaClassifier:
    """The RoBERTa classifier for classification inference"""

    def __init__(self, model_path: str = None, seed: int = 42):
        # Set the random seeds
        set_seed(seed)
        
        self.config = Config()
        self.model_path = Path(model_path) if model_path else self.config.roberta_save_dir / "final_model_roberta_lora"
        self.templates = PipelineClassificationTemplates()
        
        # Load the model and tokenizer
        self._load_model_and_tokenizer()
        
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
        
        # Move the model to GPU and set it to evaluation mode
        self.model = self.model.cuda()
        self.model.eval()
        
        # Disable dropout and other random behaviors
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm)):
                module.eval()
        
    def classify(self, question: str, schema: dict) -> tuple[int, dict]:
        """Perform classification prediction, returning the predicted class and probability distribution"""
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
    classifier = RoBERTaClassifier(
        model_path="/data/zhuyizhang/saves/RoBERTa-router/final_model_roberta_lora",
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