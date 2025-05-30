from typing import Dict, Any, List
import asyncio
from datetime import datetime
import json
import os
from tqdm import tqdm
from src.core.llm import LLMBase
from src.core.utils import load_json
from src.modules.schema_linking.base import SchemaLinkerBase
from src.modules.sql_generation.base import SQLGeneratorBase
from src.modules.post_processing.base import PostProcessorBase
from src.core.intermediate import IntermediateResult
from src.core.logger import LoggerManager
from src.core.schema.manager import SchemaManager
from concurrent.futures import ThreadPoolExecutor
from src.core.config import Config

class ElephantSQLPipeline:

    """The complete processing pipeline"""
    
    def __init__(self,
                 schema_linker: SchemaLinkerBase,
                 sql_generator: SQLGeneratorBase,
                 post_processor: PostProcessorBase,
                 pipeline_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")):
        
        self.pipeline_id = pipeline_id
        
        # Initialize the logger manager
        self.logger_manager = LoggerManager(self.pipeline_id)
        self.logger = self.logger_manager.get_logger("pipeline")
        
        # Save the module references first
        self.schema_linker = schema_linker
        self.sql_generator = sql_generator
        self.post_processor = post_processor
        
        # Set the pipeline_id and logger for each module
        self.schema_linker.intermediate = IntermediateResult(self.schema_linker.name, self.pipeline_id)
        self.schema_linker.logger = self.logger_manager.get_logger(self.schema_linker.name)
        
        self.sql_generator.intermediate = IntermediateResult(self.sql_generator.name, self.pipeline_id)
        self.sql_generator.logger = self.logger_manager.get_logger(self.sql_generator.name)
        
        self.post_processor.intermediate = IntermediateResult(self.post_processor.name, self.pipeline_id)
        self.post_processor.logger = self.logger_manager.get_logger(self.post_processor.name)
        
        # Set the dependencies between modules
        self.sql_generator.set_previous_module(self.schema_linker)
        if hasattr(self.sql_generator, 'generators'):  # Check if it is a router
            for generator in self.sql_generator.generators.values():
                generator.set_previous_module(self.schema_linker)
                
        self.post_processor.set_previous_module(self.sql_generator)
        
        # Create a pipeline-level intermediate result processor
        self.intermediate = IntermediateResult("pipeline", self.pipeline_id)
        
        # Add a statistics logger
        self.stats_logger = self.logger_manager.get_logger("api_statistics")
        
        self.schema_manager = SchemaManager()
        
    def prepare_queries(self, data_file: str) -> List[Dict]:
        """
        Prepare query data
        
        :param data_file: The path of the data file
            
        :return: List[Dict]: The prepared query list
        """
        # Load data
        merge_dev_demo_data = load_json(data_file)
        
        # Prepare batch processing data
        queries = []
        for item in merge_dev_demo_data:
            question_id = item.get("question_id", "")
            if not question_id:
                self.logger.warning("The question is missing an ID")
                continue
                
            source = item.get("source", "")
            db_id = item.get("db_id", "")
            
            # Build the database path
            # For example: ./data/merged_databases/spider_dev_academic/academic.sqlite
            db_folder = f"{source}_{db_id}"  # For example: spider_dev_academic
            db_file = f"{db_id}.sqlite"      # For example: academic.sqlite
            db_path = str(Config().database_dir / db_folder / db_file)
            
            try:
                schema = self.schema_manager.get_schema(db_id, db_path)
            except Exception as e:
                self.logger.error(f"Failed to load the database schema: {db_path}")
                self.logger.error(f"Error information: {str(e)}")
                continue
            
            queries.append({
                "query_id": question_id,
                "query": item.get("question"),
                "database_schema": schema.to_dict(),  # Convert to a dictionary format
                "source": source
            })
            
        return queries
        
    async def run_pipeline_parallel(self, data_file: str, max_workers: int = 5) -> None:
        """
        Run the complete processing pipeline in parallel

        :param data_file: The path of the data file
        :param max_workers: The maximum number of parallel workers, default is 5
        """
        start_time = datetime.now()
        self.logger.info(f"Start processing data file: {data_file}")
        
        # Set the data_file
        self.schema_linker.set_data_file(data_file)
        self.sql_generator.set_data_file(data_file)
        if hasattr(self.sql_generator, 'generators'):  # Check if it is a router
            for generator in self.sql_generator.generators.values():
                generator.set_data_file(data_file)
                
        self.post_processor.set_data_file(data_file)

        # Set the number of parallel workers
        self.intermediate.set_max_workers(max_workers)
        
        # Prepare data
        queries = self.prepare_queries(data_file)
        total_queries = len(queries)
        self.logger.info(f"Loaded {total_queries} queries to process")
        
        # Use parallel processing
        results = await self.process_batch_parallel(
            queries=queries,
            desc="Processing",
            max_workers=max_workers
        )
        
        # Calculate the total duration
        end_time = datetime.now()
        duration = end_time - start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        # Print the statistics information
        self.logger.info("="*50)
        self.logger.info("Pipeline completed")
        self.logger.info(f"Total queries: {total_queries}")
        self.logger.info(f"Maximum parallel number: {max_workers}")
        self.logger.info(f"Successfully processed: {len(results)}, Failed: {total_queries - len(results)}, Success rate: {(len(results) / total_queries * 100):.2f}%")
        self.logger.info(f"Total duration: {hours} Hours {minutes} Minutes {seconds} Seconds, Average time per query: {duration.total_seconds() / total_queries:.2f} Seconds")
        self.logger.info("="*50)
        self.log_api_stats()
    
    async def run_pipeline(self, data_file: str) -> None:
        """
        Run the complete processing pipeline in serial

        :param data_file: The path of the data file
        """
        start_time = datetime.now()
        self.logger.info(f"Start processing data file: {data_file}")
        
        # Set the data_file
        self.sql_generator.set_data_file(data_file)

        # Prepare data
        queries = self.prepare_queries(data_file)
        total_queries = len(queries)
        self.logger.info(f"Loaded {total_queries} queries to process")
        
        # Use serial processing
        results = await self.process_batch(
            queries=queries,
            desc="Processing"
        )
        
        # Calculate the total duration
        end_time = datetime.now()
        duration = end_time - start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        # Print the statistics information
        self.logger.info("="*50)
        self.logger.info("Pipeline completed")
        self.logger.info(f"Total queries: {total_queries}")
        self.logger.info(f"Successfully processed: {len(results)}, Failed: {total_queries - len(results)}, Success rate: {(len(results) / total_queries * 100):.2f}%")
        self.logger.info(f"Total duration: {hours} Hours {minutes} Minutes {seconds} Seconds, Average time per query: {duration.total_seconds() / total_queries:.2f} Seconds")
        self.logger.info("="*50)
        self.log_api_stats()
        
    def log_api_stats(self):
        """Print the API call statistics information"""
        stats_file = os.path.join(self.intermediate.pipeline_dir, "api_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                
            # Only output the brief statistics in the command line
            self.logger.warning(f"LLM API total calls: {stats['total_calls']}")
            self.logger.warning(f"Total input tokens: {stats['total_cost']['input_tokens']}; Total output tokens: {stats['total_cost']['output_tokens']}")
            
            # Display processing mode and duration information
            if stats.get('max_workers') is not None and stats['max_workers'] > 1:
                self.logger.warning(f"Processing mode: Parallel (max_workers={stats['max_workers']})")
            else:
                self.logger.warning(f"Processing mode: Serial")
                
            # Display total duration information if available
            if 'total_duration_seconds' in stats:
                total_seconds = stats['total_duration_seconds']
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                self.logger.warning(f"Total API duration: {hours} Hours {minutes} Minutes {seconds} Seconds")
            
            # Detailed statistics information written to the statistics log
            self.stats_logger.info("="*50)
            self.stats_logger.info("Detailed API call statistics")
            self.stats_logger.info("="*50)
            
            # Log processing mode
            if stats.get('max_workers') is not None and stats['max_workers'] > 1:
                self.stats_logger.info(f"Processing mode: Parallel (max_workers={stats['max_workers']})")
            else:
                self.stats_logger.info(f"Processing mode: Serial")
            
            # Log total duration information if available
            if 'total_duration_seconds' in stats:
                total_seconds = stats['total_duration_seconds']
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                self.stats_logger.info(f"Total API duration: {hours} Hours {minutes} Minutes {seconds} Seconds")
            
            self.stats_logger.info("Model statistics:")
            for model, model_stats in stats['models'].items():
                self.stats_logger.info(f"{model}:")
                self.stats_logger.info(f"  Calls: {model_stats['calls']}")
                self.stats_logger.info(f"  Input tokens: {model_stats['input_tokens']}")
                self.stats_logger.info(f"  Output tokens: {model_stats['output_tokens']}")
                self.stats_logger.info(f"  Total tokens: {model_stats['total_tokens']}")
            
            self.stats_logger.info("Module statistics:")
            for module, module_stats in stats['modules'].items():
                self.stats_logger.info(f"{module}:")
                self.stats_logger.info(f"  Total Calls: {module_stats['calls']}")
                for model, model_stats in module_stats['models'].items():
                    self.stats_logger.info(f"  {model}:")
                    self.stats_logger.info(f"    Calls: {model_stats['calls']}")
                    self.stats_logger.info(f"    Total tokens: {model_stats['total_tokens']}")
            self.stats_logger.info("="*50)
        
    async def process(self, 
                    query: str, 
                    database_schema: Dict,
                    query_id: str,
                    source: str = "") -> Dict[str, Any]:
        """
        Process a single query
        
        :param query: User query
        :param database_schema: Database schema
        :param query_id: Query ID
        :param source: Query source
            
        :return: Dict[str, Any]: The processing result
        """
        # Add source to database_schema
        database_schema["source"] = source
        
        # Add a separator at the beginning of each example
        for module in [self.schema_linker, self.sql_generator, self.post_processor]:
            module.logger.info("="*70)
            module.logger.info(f"Processing Query ID: {query_id}")
            module.logger.info(f"Query: {query}")
            
        # 1. Schema Linking with retry
        enriched_linked_schema = await self.schema_linker.link_schema_with_retry(
            query, 
            database_schema,
            query_id=query_id
        )
        
        schema_linking_output = {
            "original_schema": database_schema,
            "linked_schema": enriched_linked_schema
        }
        
        # 2. SQL Generation with retry
        extracted_sql = await self.sql_generator.generate_sql_with_retry(
            query,
            schema_linking_output,
            query_id=query_id
        )
        
        # 3. Post Processing with retry
        processed_sql = await self.post_processor.process_sql_with_retry(
            extracted_sql,
            query_id=query_id
        )
        
        # Save the final SQL result
        self.intermediate.save_sql_result(query_id, source, processed_sql)
        
        return {
            "query_id": query_id,
            "query": query,
            # "raw_linking_output": raw_linking_output,
            # "raw_sql_output": raw_sql_output,
            # "raw_postprocessing_output": raw_postprocessing_output,
            "extracted_schema": enriched_linked_schema,
            "extracted_sql": extracted_sql,
            "processed_sql": processed_sql
        } 
    
    async def process_batch(self, 
                         queries: List[Dict],
                         desc: str = "Processing") -> List[Dict]:
        """Batch process the queries in the dataset"""
        results = []
        with tqdm(total=len(queries), desc=desc) as pbar:
            for query_item in queries:
                result = await self.process(
                    query=query_item["query"],
                    database_schema=query_item["database_schema"],
                    query_id=query_item["query_id"],
                    source=query_item.get("source", "")
                )
                results.append(result)
                pbar.update(1)
        return results
    
    async def process_batch_parallel(self, queries: List[Dict], desc: str = "", max_workers: int = 5) -> List[Dict]:
        """
        Parallel processing of queries in the dataset
        
        :param queries: Query list
        :param desc: Progress bar description
        :param max_workers: Maximum number of parallel processing, default is 5
            
        :return: List[Dict]: The processing result list
        """
        results = []
        completed = 0
        total = len(queries)
        
        # Use a semaphore to limit the number of concurrent processes
        semaphore = asyncio.Semaphore(max_workers)
        
        # Record the start information
        self.logger.info(f"Start parallel processing {total} queries, maximum parallel number: {max_workers}")
        
        async def process_with_semaphore(query):
            """
            Use a semaphore to limit the number of concurrent processes
            """
            async with semaphore:
                nonlocal completed
                try:
                    self.logger.info(
                        f"Start processing query [{query['query_id']}] "
                        f"({completed + 1}/{total}, current parallel task number: {len(asyncio.all_tasks()) - 1})"
                    )
                    
                    result = await self.process(
                        query=query["query"],
                        database_schema=query["database_schema"],
                        query_id=query["query_id"],
                        source=query["source"]
                    )
                    
                    completed += 1
                    self.logger.info(
                        f"Completed query [{query['query_id']}] "
                        f"({completed}/{total}, remaining: {total - completed})"
                    )
                    return result
                except Exception as e:
                    completed += 1
                    self.logger.error(
                        f"Error occurred while processing query [{query['query_id']}]: {str(e)} "
                        f"({completed}/{total}, remaining: {total - completed})"
                    )
                    return None
        
        # Create all tasks
        tasks = [process_with_semaphore(query) for query in queries]
        
        # Use tqdm to display the progress
        with tqdm(total=total, desc=f"Processing (threads: {max_workers})") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    if result is not None:
                        results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({
                        'remain': total - completed,
                        # 'success': len(results),
                        'fail': completed - len(results)
                    })
                except Exception as e:
                    self.logger.error(f"Task execution exception: {str(e)}")
                    pbar.update(1)
                    continue
        
        # Record the completion information
        success_rate = (len(results) / total) * 100
        self.logger.info(
            f"Parallel processing completed. Total: {total}, Success: {len(results)}, "
            f"Failed: {total - len(results)}, Success rate: {success_rate:.2f}%"
        )
        
        return results 