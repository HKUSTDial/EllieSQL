from typing import Dict
from pathlib import Path
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from sklearn.neighbors import KNeighborsClassifier
from ..core.utils import load_json, load_jsonl


# Extract features
def extract_features(data):
    features = []
    labels = []
    for item in data:
        schema_info = item["enhanced_linked_schema_wo_info"]
        tables = schema_info["tables"]
        num_tables = len(tables)
        num_columns = sum(len(table["columns"]) for table in tables)
        
        features.append([num_tables, num_columns])
        label = item["pipeline_type"]
        if label in ["ADVANCED", "UNSOLVED"]:
            label = "ADV_UNSOLVED"
        labels.append(label)
    
    return features, labels



class KNNClassifierRouter(RouterBase):
    """Qwen router with classification head trained by LoRA SFT"""
    
    def __init__(self, name: str = "KNNClassifierRouter", seed: int = 42, train_file_path: str = None):
        super().__init__(name)
        self.config = Config()
        self.train_file_path = Path(train_file_path) if train_file_path else Path("data/labeled/bird_train_pipeline_label.jsonl")
        self.train_data = load_jsonl(self.train_file_path)
        X_train, y_train = extract_features(self.train_data)

        print("Training KNN classifier...")
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier.fit(X_train, y_train)
        print("Training completed")
        

    def _predict(self, question: str, schema: dict) -> tuple[int, dict]:
        """Use KNN model to predict"""
        num_tables = len(schema["tables"])
        num_columns = sum(len(table["columns"]) for table in schema["tables"])
        
        feature_vector = [[num_tables, num_columns]]
        ans = self.knn_classifier.predict(feature_vector)[0]

        if(ans == "BASIC"):
            return 0
        elif(ans == "INTERMEDIATE"):
            return 1
        else:
            return 2
        
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