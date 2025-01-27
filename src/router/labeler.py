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

class PipelineType(Enum):
    """Pipeline类型枚举"""
    VANILLA = 1             # Vanilla Pipeline 
    DC_REFINER = 2          # Divide and Conquer + Refiner
    DC_OS_REFINER = 3       # Divide and Conquer + Online Synthesis + Refiner

class Labeler:
    """用于标注数据集中每个样例对pipeline的偏好的标注器"""
    
    def __init__(self, llm: LLMBase):
        self.llm = llm
        self.extractor = TextExtractor()  # 创建一个extractor实例
        
        # 创建共用的schema linker
        def create_schema_linker():
            return EnhancedSchemaLinker(
                llm,
                model="gpt-4o-mini-2024-07-18",
                temperature=0.0,
                max_tokens=10000
            )
        
        # Pipeline 1: 简单的GPT生成
        self.vanilla_pipeline = ElephantSQLPipeline(
            schema_linker=create_schema_linker(),
            sql_generator=GPTSQLGenerator(
                llm,
                model="gpt-4o-mini-2024-07-18",
                temperature=0.0,
                max_tokens=10000
            ),
            post_processor=SkipPostProcessor()
        )
        
        # Pipeline 2: 使用DC Refiner
        self.dc_refiner_pipeline = ElephantSQLPipeline(
            schema_linker=create_schema_linker(),
            sql_generator=DCRefinerSQLGenerator(
                llm,
                model="gpt-4o-mini-2024-07-18",
                temperature=0.0,
                max_tokens=10000
            ),
            post_processor=SkipPostProcessor()
        )
        
        # Pipeline 3: 使用Enhanced生成器
        self.enhanced_pipeline = ElephantSQLPipeline(
            schema_linker=create_schema_linker(),
            sql_generator=EnhancedSQLGenerator(
                llm,
                model="gpt-4o-mini-2024-07-18",
                temperature=0.0,
                max_tokens=10000
            ),
            post_processor=SkipPostProcessor()
        )
        
        # 保存pipeline映射关系
        self.pipelines = {
            PipelineType.VANILLA: self.vanilla_pipeline,
            PipelineType.DC_REFINER: self.dc_refiner_pipeline,
            PipelineType.DC_OS_REFINER: self.enhanced_pipeline
        }

    async def label_dataset(self, data_file: str, output_file: str, max_workers: int = 5):
        """标注数据集"""
        # 加载数据集
        dataset = load_json(data_file)
        labeled_results = []
        
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 设置数据文件路径给各个模块
        for pipeline in self.pipelines.values():
            pipeline.schema_linker.set_data_file(data_file)
            pipeline.sql_generator.set_data_file(data_file)
            pipeline.post_processor.set_data_file(data_file)
        
        # 对每个样例进行标注
        for item in tqdm(dataset, desc="Labeling dataset"):
            try:
                query_id = item["question_id"]
                source = item["source"]
                db_id = item["db_id"]
                gold_sql = item["gold_SQL"]
                db_path = f"data/merged_databases/{source}_{db_id}/{db_id}.sqlite"
                
                # label默认值使用最复杂的pipeline：即如果较为简单的pipeline不能生成正确的SQL，则label为最复杂的Pipeline 3
                label = PipelineType.DC_OS_REFINER.value
                
                # 先获取schema linking结果
                linked_schema_raw = await self.vanilla_pipeline.schema_linker.link_schema(
                    query=item["question"],
                    database_schema=self.vanilla_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                    query_id=query_id
                )
                
                # 获取仅包含主键外键的enhanced schema
                enhanced_linked_schema_wo_info = self.vanilla_pipeline.schema_linker.enhance_schema_only_with_keys(
                    json.loads(self.extractor.extract_code_block(linked_schema_raw, 'json')),
                    self.vanilla_pipeline.schema_manager.get_schema(db_id, db_path).to_dict()
                )
                
                # 尝试使用Pipeline 1: Vanilla Pipeline
                result1 = await self.vanilla_pipeline.process(
                    query=item["question"],
                    database_schema=self.vanilla_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                    query_id=query_id,
                    source=source
                )
                if result1:
                    generated_sql = result1.get("generated_sql", "")
                    result = compare_sql_results(db_path, gold_sql, generated_sql)
                    is_correct = result[0] if isinstance(result, tuple) else result
                    
                    if is_correct:
                        label = PipelineType.VANILLA.value
                    else:
                        # 尝试使用Pipeline 2: DC Refiner Pipeline
                        result2 = await self.dc_refiner_pipeline.process(
                            query=item["question"],
                            database_schema=self.vanilla_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                            query_id=query_id,
                            source=source
                        )
                        if result2:
                            generated_sql = result2.get("generated_sql", "")
                            result = compare_sql_results(db_path, gold_sql, generated_sql)
                            is_correct = result[0] if isinstance(result, tuple) else result
                            
                            if is_correct:
                                label = PipelineType.DC_REFINER.value
                
                # 保存标注结果
                labeled_result = {
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
                labeled_results.append(labeled_result)
                
                # 写入文件
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(labeled_result, ensure_ascii=False) + "\n")
                    
            except Exception as e:
                print(f"Error processing question {query_id}: {str(e)}")
                continue
                
        return labeled_results

async def main():
    # 初始化LLM
    llm = LLMBase()
    
    # 创建标注器
    labeler = Labeler(llm)
    
    # 标注数据集
    await labeler.label_dataset(
        data_file="./data/merge_dev_demo.json",
        output_file="./data/labeled/merge_dev_demo_pipeline_preference.jsonl",
        max_workers=5
    )

if __name__ == "__main__":
    asyncio.run(main()) 