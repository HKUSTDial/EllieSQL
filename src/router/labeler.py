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
from ..evaluation.compute_ex import compare_sql_results
from ..core.utils import load_json, TextExtractor
from ..pipeline_factory import PipelineFactory, PipelineLevel
from ..core.config import Config
from ..core.logger import LoggerManager

class PipelineType(Enum):
    """Pipeline type enumeration"""
    BASIC = 1           # Can be processed by Basic pipeline
    INTERMEDIATE = 2    # Needs to be processed by Intermediate pipeline
    ADVANCED = 3        # Needs to be processed by Advanced pipeline
    UNSOLVED = 4        # All pipelines cannot be processed, but actually considered ADVANCED in the training set

class Labeler:
    """The labeler for labeling the preference of each example in the training set for the pipeline"""
    
    def __init__(self, llm: LLMBase):
        self.llm = llm
        self.extractor = TextExtractor()
        self.pipeline_factory = PipelineFactory(llm, backbone_model="gpt-3.5-turbo", temperature=0.0, max_retries=10)
        
        # Get different levels of pipelines
        self.basic_pipeline = self.pipeline_factory.get_pipeline(PipelineLevel.BASIC)
        self.intermediate_pipeline = self.pipeline_factory.get_pipeline(PipelineLevel.INTERMEDIATE)
        self.advanced_pipeline = self.pipeline_factory.get_pipeline(PipelineLevel.ADVANCED)

        # Initialize logs
        self.logger_manager = LoggerManager()
        self.logger = self.logger_manager.get_logger("labeler")
        self.stats_logger = self.logger_manager.get_logger("labeling_stats")

    async def label_single_item(self, item: Dict, data_file: str) -> Dict:
        """Label a single example, try in the order of increasing pipeline difficulty, until finding the pipeline that can correctly process it"""
        query_id = item["question_id"]
        source = item["source"]
        db_id = item["db_id"]
        gold_sql = item["gold_SQL"]
        db_folder = f"{source}_{db_id}"
        db_file = f"{db_id}.sqlite"
        db_path = str(Config().database_dir / db_folder / db_file)
        
        self.logger.info(f"开始处理样例 [ID: {query_id}] 数据库: {db_path}, 问题: {item['question']}")
        
        try:
            # Use schema linking with retry mechanism
            linked_schema = await self.basic_pipeline.schema_linker.link_schema_with_retry(
                query=item["question"],
                database_schema=self.basic_pipeline.schema_manager.get_schema(db_id, db_path).to_dict(),
                query_id=query_id
            )
            
            if linked_schema is None:
                self.logger.error(f"[ID: {query_id}] Schema linking失败")
                return None
            
            # Get enhanced schema
            enhanced_linked_schema_wo_info = self.basic_pipeline.schema_linker.enhance_schema_only_with_keys(
                linked_schema,  # Use the returned dictionary directly
                self.basic_pipeline.schema_manager.get_schema(db_id, db_path).to_dict()
            )

            # 1. Try BASIC Pipeline
            self.logger.debug(f"[ID: {query_id}] Try BASIC Pipeline...")
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

            # 2. BASIC failed, try INTERMEDIATE Pipeline
            self.logger.debug(f"[ID: {query_id}] BASIC failed, try INTERMEDIATE Pipeline...")
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

            # 3. INTERMEDIATE failed, try ADVANCED Pipeline
            self.logger.debug(f"[ID: {query_id}] INTERMEDIATE failed, try ADVANCED Pipeline...")
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

            # 4. All pipelines failed
            self.logger.warning(f"[ID: {query_id}] All pipelines failed")
            label = PipelineType.UNSOLVED.value
            return self._create_result_dict(item, label, enhanced_linked_schema_wo_info)
            
        except Exception as e:
            self.logger.error(f"[ID: {query_id}] Error occurred while processing the example: {str(e)}")
            return None

    def _create_result_dict(self, item: Dict, label: int, enhanced_schema: Dict) -> Dict:
        """Create the result dictionary"""
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
        """Label the dataset"""
        # Use the config path
        data_file = str(Config().data_dir / Path(data_file))
        output_file = str(Config().data_dir / Path(output_file))
        
        self.logger.info(f"Start processing data file: {data_file}")
        self.logger.info(f"Output file: {output_file}")
        
        # Load the dataset
        dataset = load_json(data_file)
        labeled_results = []
        
        # Create the output directory
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set the data file path for each module
        for pipeline in [self.basic_pipeline, self.intermediate_pipeline, self.advanced_pipeline]:
            pipeline.schema_linker.set_data_file(data_file)
            pipeline.sql_generator.set_data_file(data_file)
            pipeline.post_processor.set_data_file(data_file)
        
        # Create the task list
        tasks = []
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await self.label_single_item(item, data_file)
        
        # Create all tasks
        for item in dataset:
            tasks.append(process_with_semaphore(item))
        
        # Use tqdm to show progress
        with tqdm(total=len(tasks), desc=f"Labeling dataset (threads: {max_workers})") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    labeled_results.append(result)
                    # Write to file
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                pbar.update(1)
                
        # Record labeling statistics
        self._log_labeling_stats(labeled_results)
        return labeled_results

    def _log_labeling_stats(self, results: List[Dict]):
        """Record labeling statistics"""
        total = len(results)
        label_counts = {label.name: 0 for label in PipelineType}
        source_stats = {}
        difficulty_stats = {}
        
        for result in results:
            # Count the label distribution
            pipeline_type = result["pipeline_type"]
            label_counts[pipeline_type] += 1
            
            # Count the source distribution
            source = result["source"]
            if source not in source_stats:
                source_stats[source] = {label.name: 0 for label in PipelineType}
            source_stats[source][pipeline_type] += 1
            
            # Count the difficulty distribution
            if "difficulty" in result:
                difficulty = result["difficulty"]
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {label.name: 0 for label in PipelineType}
                difficulty_stats[difficulty][pipeline_type] += 1
        
        # Prepare statistics
        stats_lines = [
            "="*50,
            "Labeling statistics",
            "="*50,
            f"Total samples: {total}",
            "\nLabel distribution:"
        ]
        
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            stats_lines.append(f"{label}: {count} ({percentage:.2f}%)")
            
        stats_lines.append("\nSource distribution:")
        for source, stats in source_stats.items():
            stats_lines.append(f"\n{source}:")
            source_total = sum(stats.values())
            for label, count in stats.items():
                percentage = (count / source_total) * 100
                stats_lines.append(f"  {label}: {count} ({percentage:.2f}%)")
                
        if difficulty_stats:
            stats_lines.append("\nDifficulty distribution:")
            for difficulty, stats in difficulty_stats.items():
                stats_lines.append(f"\n{difficulty}:")
                diff_total = sum(stats.values())
                for label, count in stats.items():
                    percentage = (count / diff_total) * 100
                    stats_lines.append(f"  {label}: {count} ({percentage:.2f}%)")
        
        stats_lines.append("="*50)
        
        # Record to log and print to console
        stats_text = "\n".join(stats_lines)
        self.stats_logger.info(stats_text)
        print("\n" + stats_text)

async def main():
    # Initialize LLM
    llm = LLMBase()
    
    # Create the labeler
    labeler = Labeler(llm)
    
    # Label the dataset
    await labeler.label_dataset_parallel(
        data_file="formatted_bird_dev.json",
        output_file="labeled/bird_dev_pipeline_label.jsonl",
        max_workers=100
    )

if __name__ == "__main__":
    # 1. Set a new event loop
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())

    # 2. Set the thread pool size
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=300))

    # 3. Run and close
    loop.run_until_complete(main())
    loop.close()
