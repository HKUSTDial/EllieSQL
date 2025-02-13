from typing import Dict
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from sklearn.neighbors import KNeighborsClassifier
from ..core.utils import load_json, load_jsonl


    # 提取特征
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
    """添加了分类头并使用LoRA SFT的Qwen路由器"""
    
    def __init__(self, name: str = "KNNClassifierRouter", seed: int = 42):
        super().__init__(name)
        self.config = Config()
        self.train_file_path = "./data/labeled/nonempty_bird_train_pipeline_label.jsonl"
        self.train_data = load_jsonl(self.train_file_path)
        X_train, y_train = extract_features(self.train_data)


        print("正在训练KNN分类器。。。")
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier.fit(X_train, y_train)
        print("训练完成")
        


    def _predict(self, question: str, schema: dict) -> tuple[int, dict]:
        """使用KNN模型进行预测"""


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