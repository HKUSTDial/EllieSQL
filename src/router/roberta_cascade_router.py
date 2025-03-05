import torch
from typing import Dict
from pathlib import Path
from .base import RouterBase
from ..pipeline_factory import PipelineLevel
from ..core.config import Config
from ..sft.roberta_cascade_inference import RoBERTaCascadeClassifier

class RoBERTaCascadeRouter(RouterBase):
    """级联式RoBERTa路由器：按basic->intermediate->advanced顺序判断"""
    
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
        
        # 初始化pipeline对应的分类器
        self.basic_classifier = RoBERTaCascadeClassifier(
            pipeline_type="basic",
            model_name="basic_classifier",
            confidence_threshold=confidence_threshold,
            seed=seed
        )
        print("Successfully load and initialize classifier for basic pipeline")
        
        self.intermediate_classifier = RoBERTaCascadeClassifier(
            pipeline_type="intermediate",
            model_name="intermediate_classifier",
            confidence_threshold=confidence_threshold,
            seed=seed
        )
        print("Successfully load and initialize classifier for intermediate pipeline")

        # intermediate无法处理的直接使用advanced, 不需要额外判断
        # self.advanced_classifier = RoBERTaCascadeClassifier(
        #     pipeline_type="advanced",
        #     model_name="advanced_classifier",
        #     confidence_threshold=confidence_threshold,
        #     seed=seed
        # )
        # print("Successfully load and initialize classifier for advanced pipeline")
        
        # 记录每个pipeline的处理函数
        self.pipeline_handlers = {}
        
    async def route(self, query: str, schema_linking_output: Dict, query_id: str) -> str:
        """级联式路由逻辑"""
        linked_schema = schema_linking_output.get("linked_schema", {})
        
        # 1. 首先尝试basic pipeline
        can_handle, confidence = self.basic_classifier.classify(query, linked_schema)
        if can_handle:
            self.logger.info(f"Query {query_id} routed to BASIC pipeline (confidence: {confidence:.3f})")
            return PipelineLevel.BASIC.value
            
        # 2. 如果basic不行，尝试intermediate pipeline
        can_handle, confidence = self.intermediate_classifier.classify(query, linked_schema)
        if can_handle:
            self.logger.info(f"Query {query_id} routed to INTERMEDIATE pipeline (confidence: {confidence:.3f})")
            return PipelineLevel.INTERMEDIATE.value
            
        # 3. 如果intermediate不行，直接使用advanced pipeline
        # 可选：也可以先检查advanced pipeline是否能处理
        # can_handle, confidence = self.advanced_classifier.classify(query, linked_schema)
        self.logger.info(
            f"Query {query_id} routed to ADVANCED pipeline (confidence: {confidence:.3f})"
            # f"({'can handle' if can_handle else 'cannot handle'}, confidence: {confidence:.3f})"
        )
        return PipelineLevel.ADVANCED.value
        