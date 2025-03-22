import asyncio
import argparse
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from .core.llm import LLMBase
from .modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from .modules.sql_generation.gpt_generator import GPTSQLGenerator
from .modules.sql_generation.enhanced_generator import EnhancedSQLGenerator
from .modules.sql_generation.online_synthesis_refiner_generator import OSRefinerSQLGenerator
from .modules.post_processing.skip_post_processing import SkipPostProcessor
from .pipeline import ElephantSQLPipeline
from src.router.table_count_router import TableCountRouter
from src.router.knn_classifier_router import KNNClassifierRouter
from src.router.qwen_classifier_router import QwenClassifierRouter
from src.router.qwen_pairwise_router import QwenPairwiseRouter
from src.router.roberta_classifier_router import RoBERTaClassifierRouter
from src.router.roberta_cascade_router import RoBERTaCascadeRouter
from src.router.qwen_cascade_router import QwenCascadeRouter
from src.router.dpo_qwen_router import DPOClassifierRouter
from src.pipeline_factory import PipelineFactory, PipelineLevel

def get_router(router_type: str, router_path: str = None, confidence_threshold: float = 0.6, train_file_path: str = None):
    """根据参数获取对应的router"""
    if router_type == "knn":
        return KNNClassifierRouter(
            seed=42,
            train_file_path=train_file_path
        )
    elif router_type == "qwen":
        return QwenClassifierRouter(
            seed=42,
            lora_path=router_path
        )
    elif router_type == "roberta":
        return RoBERTaClassifierRouter(
            seed=42,
            model_path=router_path
        )
    elif router_type == "qwen_cascade":
        return QwenCascadeRouter(
            seed=42,
            confidence_threshold=confidence_threshold,
            model_path=router_path
        )
    elif router_type == "roberta_cascade":
        return RoBERTaCascadeRouter(
            seed=42,
            confidence_threshold=confidence_threshold,
            model_path=router_path
        )
    elif router_type == "qwen_pairwise":
        return QwenPairwiseRouter(
            seed=42,
            lora_path=router_path
        )
    elif router_type == "qwen_dpo":
        return DPOClassifierRouter(
            seed=42,
            model_path=router_path
        )
    else:
        raise ValueError(f"Unknown router type: {router_type}")

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--router', type=str, required=True,
                       choices=['knn', 'qwen', 'roberta', 'qwen_cascade', 
                               'roberta_cascade', 'qwen_pairwise', 'qwen_dpo'],
                       help='Type of router to use')
    parser.add_argument('--router_path', type=str, default=None,
                       help='Path to router model/weights')
    parser.add_argument('--train_file_path', type=str, default=None,
                       help='Path to train file')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Confidence threshold for cascade routers')
    parser.add_argument('--backbone_model', type=str, default="gpt-4o-mini-2024-07-18",
                       help='Backbone LLM model to use')
    parser.add_argument('--test_file', type=str, default="./data/formatted_bird_dev.json",
                       help='Test data file path')
    parser.add_argument('--max_workers', type=int, default=16,
                       help='Maximum number of parallel workers')
    args = parser.parse_args()

    # 初始化LLM
    llm = LLMBase()
    
    # 创建pipeline工厂
    factory = PipelineFactory(
        llm, 
        backbone_model=args.backbone_model,
        temperature=0.0,
        max_retries=10
    )

    # 获取指定的router
    router = get_router(
        args.router,
        args.router_path,
        args.confidence_threshold,
        args.train_file_path
    )
    
    # 注册生成器
    router.register_generator(PipelineLevel.BASIC.value, 
                            factory.get_pipeline(PipelineLevel.BASIC).sql_generator)
    router.register_generator(PipelineLevel.INTERMEDIATE.value, 
                            factory.get_pipeline(PipelineLevel.INTERMEDIATE).sql_generator)
    router.register_generator(PipelineLevel.ADVANCED.value, 
                            factory.get_pipeline(PipelineLevel.ADVANCED).sql_generator)
    
    # 创建使用router的pipeline
    pipeline = ElephantSQLPipeline(
        schema_linker=factory.get_pipeline(PipelineLevel.BASIC).schema_linker, # 复用基础pipeline的schema linker
        sql_generator=router, # 使用router代替特定的generator
        post_processor=SkipPostProcessor()
    )
    
    # 运行pipeline
    await pipeline.run_pipeline_parallel(
        data_file=args.test_file,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    # 1. 设置新的事件循环
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    # 2. 设置线程池大小
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=256))

    # 3. 运行与关闭
    loop.run_until_complete(main())
    loop.close()
