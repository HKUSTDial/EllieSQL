import asyncio
from typing import Dict, List, Optional
import json
from pathlib import Path
from enum import Enum
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
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
from ..core.config import Config
from ..core.logger import LoggerManager

class PipelineType(Enum):
    """Pipeline类型枚举"""
    BASIC = 1           # 可以使用简单Pipeline处理
    INTERMEDIATE = 2    # 需要使用中等Pipeline处理
    ADVANCED = 3        # 需要使用高级Pipeline处理
    UNSOLVED = 4        # 所有Pipeline都无法处理, 但是training set中实际看作ADVANCED标签

class Labeler:
    """用于标注训练集中每个样例对pipeline的偏好的标注器"""
    
    def __init__(self, llm: LLMBase):
        self.llm = llm
        self.extractor = TextExtractor()
        self.pipeline_factory = PipelineFactory(llm, backbone_model="gpt-3.5-turbo", temperature=0.0, max_retries=10)
        
        # 获取不同级别的pipeline
        self.basic_pipeline = self.pipeline_factory.get_pipeline(PipelineLevel.BASIC)
        self.intermediate_pipeline = self.pipeline_factory.get_pipeline(PipelineLevel.INTERMEDIATE)
        self.advanced_pipeline = self.pipeline_factory.get_pipeline(PipelineLevel.ADVANCED)

        # 初始化日志
        self.logger_manager = LoggerManager()
        self.logger = self.logger_manager.get_logger("labeler")
        self.stats_logger = self.logger_manager.get_logger("labeling_stats")

    async def label_single_item(self, item: Dict, data_file: str) -> Dict:
        """标注单个样例,按照pipeline难度递增的顺序尝试,直到找到能正确处理的pipeline"""
        query_id = item["question_id"]
        source = item["source"]
        db_id = item["db_id"]
        gold_sql = item["gold_SQL"]
        db_folder = f"{source}_{db_id}"
        db_file = f"{db_id}.sqlite"
        db_path = str(Config().database_dir / db_folder / db_file)
        
        self.logger.info(f"开始处理样例 [ID: {query_id}] 数据库: {db_path}, 问题: {item['question']}")
        
        try:
            # 使用带重试机制的schema linking
            linked_schema = await self.basic_pipeline.schema_linker.link_schema_with_retry(
                query=item["question"],
                database_schema=self.basic_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                query_id=query_id
            )
            
            if linked_schema is None:
                self.logger.error(f"[ID: {query_id}] Schema linking失败")
                return None
            
            # 获取enhanced schema
            enhanced_linked_schema_wo_info = self.basic_pipeline.schema_linker.enhance_schema_only_with_keys(
                linked_schema,  # 直接使用返回的字典
                self.basic_pipeline.schema_manager.get_schema(db_id, db_path).to_dict()
            )

            # 1. 尝试BASIC Pipeline
            self.logger.debug(f"[ID: {query_id}] 尝试BASIC Pipeline...")
            result = await self.basic_pipeline.process(
                query=item["question"],
                database_schema=self.basic_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                query_id=query_id,
                source=source
            )
            if result:
                processed_sql = result.get("processed_sql", "")
                check_sql_result = compare_sql_results(db_path, gold_sql, processed_sql)
                is_correct = check_sql_result[0] if isinstance(check_sql_result, tuple) else check_sql_result
                
                if is_correct:
                    self.logger.info(f"[ID: {query_id}] BASIC Pipeline成功处理")
                    label = PipelineType.BASIC.value
                    return self._create_result_dict(item, label, enhanced_linked_schema_wo_info)

            # 2. BASIC失败,尝试INTERMEDIATE Pipeline
            self.logger.debug(f"[ID: {query_id}] BASIC失败, 尝试INTERMEDIATE Pipeline...")
            result = await self.intermediate_pipeline.process(
                query=item["question"],
                database_schema=self.intermediate_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                query_id=query_id,
                source=source
            )
            if result:
                processed_sql = result.get("processed_sql", "")
                check_sql_result = compare_sql_results(db_path, gold_sql, processed_sql)
                is_correct = check_sql_result[0] if isinstance(check_sql_result, tuple) else check_sql_result
                
                if is_correct:
                    self.logger.info(f"[ID: {query_id}] INTERMEDIATE Pipeline成功处理")
                    label = PipelineType.INTERMEDIATE.value
                    return self._create_result_dict(item, label, enhanced_linked_schema_wo_info)

            # 3. INTERMEDIATE失败,尝试ADVANCED Pipeline
            self.logger.debug(f"[ID: {query_id}] INTERMEDIATE失败, 尝试ADVANCED Pipeline...")
            result = await self.advanced_pipeline.process(
                query=item["question"],
                database_schema=self.advanced_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                query_id=query_id,
                source=source
            )
            if result:
                processed_sql = result.get("processed_sql", "")
                check_sql_result = compare_sql_results(db_path, gold_sql, processed_sql)
                is_correct = check_sql_result[0] if isinstance(check_sql_result, tuple) else check_sql_result
                
                if is_correct:
                    self.logger.info(f"[ID: {query_id}] ADVANCED Pipeline成功处理")
                    label = PipelineType.ADVANCED.value
                    return self._create_result_dict(item, label, enhanced_linked_schema_wo_info)

            # 4. 所有Pipeline都失败
            self.logger.warning(f"[ID: {query_id}] 所有Pipeline都失败")
            label = PipelineType.UNSOLVED.value
            return self._create_result_dict(item, label, enhanced_linked_schema_wo_info)
            
        except Exception as e:
            self.logger.error(f"[ID: {query_id}] 处理样例时发生错误: {str(e)}")
            return None

    def _create_result_dict(self, item: Dict, label: int, enhanced_schema: Dict) -> Dict:
        """创建结果字典"""
        return {
            "question_id": item["question_id"],
            "source": item["source"],
            "db_id": item["db_id"],
            "question": item.get("question", ""),
            "difficulty": item.get("difficulty", ""),
            "gold_sql": item["gold_SQL"],
            "pipeline_type": PipelineType(label).name,
            "label": label,
            "enhanced_linked_schema_wo_info": enhanced_schema
        }

    async def label_dataset_parallel(self, data_file: str, output_file: str, max_workers: int = 5):
        """标注数据集"""
        # 使用配置路径
        data_file = str(Config().data_dir / Path(data_file))
        output_file = str(Config().data_dir / Path(output_file))
        
        self.logger.info(f"开始处理数据文件: {data_file}")
        self.logger.info(f"输出文件: {output_file}")
        
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
                
        # 记录标注统计信息
        self._log_labeling_stats(labeled_results)
        return labeled_results

    def _log_labeling_stats(self, results: List[Dict]):
        """记录标注统计信息"""
        total = len(results)
        label_counts = {label.name: 0 for label in PipelineType}
        source_stats = {}
        difficulty_stats = {}
        
        for result in results:
            # 统计标签分布
            pipeline_type = result["pipeline_type"]
            label_counts[pipeline_type] += 1
            
            # 按来源统计
            source = result["source"]
            if source not in source_stats:
                source_stats[source] = {label.name: 0 for label in PipelineType}
            source_stats[source][pipeline_type] += 1
            
            # 按难度统计
            if "difficulty" in result:
                difficulty = result["difficulty"]
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {label.name: 0 for label in PipelineType}
                difficulty_stats[difficulty][pipeline_type] += 1
        
        # 准备统计信息
        stats_lines = [
            "="*50,
            "标注统计信息",
            "="*50,
            f"总样本数: {total}",
            "\n标签分布:"
        ]
        
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            stats_lines.append(f"{label}: {count} ({percentage:.2f}%)")
            
        stats_lines.append("\n按来源统计:")
        for source, stats in source_stats.items():
            stats_lines.append(f"\n{source}:")
            source_total = sum(stats.values())
            for label, count in stats.items():
                percentage = (count / source_total) * 100
                stats_lines.append(f"  {label}: {count} ({percentage:.2f}%)")
                
        if difficulty_stats:
            stats_lines.append("\n按难度统计:")
            for difficulty, stats in difficulty_stats.items():
                stats_lines.append(f"\n{difficulty}:")
                diff_total = sum(stats.values())
                for label, count in stats.items():
                    percentage = (count / diff_total) * 100
                    stats_lines.append(f"  {label}: {count} ({percentage:.2f}%)")
        
        stats_lines.append("="*50)
        
        # 同时记录到日志和打印到控制台
        stats_text = "\n".join(stats_lines)
        self.stats_logger.info(stats_text)
        print("\n" + stats_text)

async def main():
    # 初始化LLM
    llm = LLMBase()
    
    # 创建标注器
    labeler = Labeler(llm)
    
    # 标注数据集
    await labeler.label_dataset_parallel(
        data_file="formatted_bird_dev.json",
        output_file="labeled/bird_dev_pipeline_label.jsonl",
        max_workers=100
    )

if __name__ == "__main__":
    # 1. 设置新的事件循环
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    # 2. 设置线程池大小
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=300))

    # 3. 运行与关闭
    loop.run_until_complete(main())
    loop.close()
