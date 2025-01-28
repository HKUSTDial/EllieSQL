import asyncio
from typing import Dict, List, Optional
import json
from pathlib import Path
from enum import Enum
from tqdm import tqdm
from ..core.llm import LLMBase
from ..modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from ..modules.sql_generation.gpt_generator import GPTSQLGenerator
from ..modules.sql_generation.dc_refiner_generator import DCRefinerSQLGenerator
from ..modules.sql_generation.enhanced_generator import EnhancedSQLGenerator
from ..modules.post_processing.skip_post_processing import SkipPostProcessor
from ..pipeline import ElephantSQLPipeline
from ..evaluation.compute_EX import compare_sql_results
from ..core.utils import load_json, TextExtractor
from ..pipeline_factory import PipelineFactory, PipelineLevel

class PipelineType(Enum):
    """Pipeline类型枚举"""
    VANILLA = 1             # Vanilla Pipeline 
    DC_REFINER = 2          # Divide and Conquer + Refiner
    DC_OS_REFINER = 3       # Divide and Conquer + Online Synthesis + Refiner

class Labeler:
    """用于标注数据集中每个样例对pipeline的偏好的标注器"""
    
    def __init__(self, llm: LLMBase):
        self.llm = llm
        self.extractor = TextExtractor()
        self.pipeline_factory = PipelineFactory(llm)
        
        # 获取不同级别的pipeline
        self.basic_pipeline = self.pipeline_factory.get_pipeline(PipelineLevel.BASIC)
        self.intermediate_pipeline = self.pipeline_factory.get_pipeline(PipelineLevel.INTERMEDIATE)
        self.advanced_pipeline = self.pipeline_factory.get_pipeline(PipelineLevel.ADVANCED)

    async def label_single_item(self, item: Dict, data_file: str) -> Dict:
        """标注单个样例"""
        query_id = item["question_id"]
        source = item["source"]
        db_id = item["db_id"]
        gold_sql = item["gold_SQL"]
        db_path = f"data/merged_databases/{source}_{db_id}/{db_id}.sqlite"
        
        # label默认值使用最复杂的pipeline, 如果简单pipeline能正确生成, 则使用简单pipeline替代标注
        label = PipelineType.DC_OS_REFINER.value
        
        try:
            # 先获取schema linking结果
            linked_schema_raw = await self.basic_pipeline.schema_linker.link_schema(
                query=item["question"],
                database_schema=self.basic_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                query_id=query_id
            )
            
            # 获取仅包含主键外键的enhanced schema
            enhanced_linked_schema_wo_info = self.basic_pipeline.schema_linker.enhance_schema_only_with_keys(
                json.loads(self.extractor.extract_code_block(linked_schema_raw, 'json')),
                self.basic_pipeline.schema_manager.get_schema(db_id, db_path).to_dict()
            )
            
            # 尝试使用Pipeline 1: Vanilla Pipeline
            result1 = await self.basic_pipeline.process(
                query=item["question"],
                database_schema=self.basic_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                query_id=query_id,
                source=source
            )
            if result1:
                processed_sql = result1.get("processed_sql", "")
                check_sql_result = compare_sql_results(db_path, gold_sql, processed_sql)
                is_correct = check_sql_result[0] if isinstance(check_sql_result, tuple) else check_sql_result
                
                if is_correct:
                    label = PipelineType.VANILLA.value
                
                else:
                    # 尝试使用Pipeline 2: DC Refiner Pipeline
                    result2 = await self.intermediate_pipeline.process(
                        query=item["question"],
                        database_schema=self.basic_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                        query_id=query_id,
                        source=source
                    )
                    if result2:
                        processed_sql = result2.get("processed_sql", "")
                        check_sql_result = compare_sql_results(db_path, gold_sql, processed_sql)
                        is_correct = check_sql_result[0] if isinstance(check_sql_result, tuple) else check_sql_result
                        
                        if is_correct:
                            label = PipelineType.DC_REFINER.value

            return {
                "question_id": query_id,
                "source": source,
                "db_id": db_id,
                "question": item.get("question", ""),
                "difficulty": item.get("difficulty", ""),
                "gold_sql": gold_sql,
                "pipeline_type": PipelineType(label).name,
                "label": label,
                "enhanced_linked_schema_wo_info": enhanced_linked_schema_wo_info
            }
            
        except Exception as e:
            print(f"Error processing question {query_id}: {str(e)}")
            return None

    async def label_dataset_parallel(self, data_file: str, output_file: str, max_workers: int = 5):
        """标注数据集"""
        # 加载数据集
        dataset = load_json(data_file)
        labeled_results = []
        
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 设置数据文件路径给各个模块
        for pipeline in [self.basic_pipeline, self.intermediate_pipeline, self.advanced_pipeline]:
            pipeline.schema_linker.set_data_file(data_file)
            pipeline.sql_generator.set_data_file(data_file)
            pipeline.post_processor.set_data_file(data_file)
        
        # 创建任务列表
        tasks = []
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await self.label_single_item(item, data_file)
        
        # 创建所有任务
        for item in dataset:
            tasks.append(process_with_semaphore(item))
        
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc=f"Labeling dataset (threads: {max_workers})") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    labeled_results.append(result)
                    # 写入文件
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                pbar.update(1)
                
        return labeled_results

async def main():
    # 初始化LLM
    llm = LLMBase()
    
    # 创建标注器
    labeler = Labeler(llm)
    
    # 标注数据集
    await labeler.label_dataset_parallel(
        data_file="./data/sampled_bird_demo.json",
        output_file="./data/labeled/sampled_bird_demo_pipeline_preference.jsonl",
        max_workers=10
    )

if __name__ == "__main__":
    asyncio.run(main()) 