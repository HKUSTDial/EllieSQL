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

class ElephantSQLPipeline:
    """ElephantSQL系统的主要pipeline"""
    
    def __init__(self,
                 schema_linker: SchemaLinkerBase,
                 sql_generator: SQLGeneratorBase,
                 post_processor: PostProcessorBase):
        self.pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化日志管理器
        self.logger_manager = LoggerManager(self.pipeline_id)
        self.logger = self.logger_manager.get_logger("pipeline")
        
        # 为每个模块设置pipeline_id和logger
        schema_linker.intermediate = IntermediateResult(schema_linker.name, self.pipeline_id)
        schema_linker.logger = self.logger_manager.get_logger(schema_linker.name)
        
        sql_generator.intermediate = IntermediateResult(sql_generator.name, self.pipeline_id)
        sql_generator.logger = self.logger_manager.get_logger(sql_generator.name)
        
        post_processor.intermediate = IntermediateResult(post_processor.name, self.pipeline_id)
        post_processor.logger = self.logger_manager.get_logger(post_processor.name)
        
        # 设置模块间的关联
        sql_generator.set_previous_module(schema_linker)
        post_processor.set_previous_module(sql_generator)
        
        self.schema_linker = schema_linker
        self.sql_generator = sql_generator
        self.post_processor = post_processor
        
        # 创建pipeline级别的中间结果处理器
        self.intermediate = IntermediateResult("pipeline", self.pipeline_id)
        
        # 添加统计日志记录器
        self.stats_logger = self.logger_manager.get_logger("api_statistics")
        
        self.schema_manager = SchemaManager()
        
    def prepare_queries(self, data_file: str) -> List[Dict]:
        """
        准备查询数据
        
        Args:
            data_file: 数据文件路径
            
        Returns:
            List[Dict]: 准备好的查询列表
        """
        # 加载数据
        merge_dev_demo_data = load_json(data_file)
        
        # 准备批处理数据
        queries = []
        for item in merge_dev_demo_data:
            question_id = item.get("question_id", "")
            if not question_id:
                self.logger.warning("问题缺少ID")
                continue
                
            source = item.get("source", "")
            db_id = item.get("db_id", "")
            
            # 构建数据库路径
            # 例如: ./data/merged_databases/spider_dev_academic/academic.sqlite
            db_folder = f"{source}_{db_id}"  # 例如: spider_dev_academic
            db_file = f"{db_id}.sqlite"      # 例如: academic.sqlite
            db_path = f"./data/merged_databases/{db_folder}/{db_file}"
            
            try:
                schema = self.schema_manager.get_schema(db_id, db_path)
            except Exception as e:
                self.logger.error(f"加载数据库schema失败: {db_path}")
                self.logger.error(f"错误信息: {str(e)}")
                continue
            
            queries.append({
                "query_id": question_id,
                "query": item.get("question"),
                "database_schema": schema.to_dict(),  # 转换为字典格式
                "source": source
            })
            
        return queries
        
    async def run_pipeline_parallel(self, data_file: str, max_workers: int = 5) -> None:
        """
        并行化运行完整的处理流程

        Args:
            data_file: 数据文件路径
            max_workers: 最大并行数，默认为5
        """
        start_time = datetime.now()
        self.logger.info(f"开始处理数据文件: {data_file}")
        
        # 设置data_file
        self.sql_generator.set_data_file(data_file)

        # 准备数据
        queries = self.prepare_queries(data_file)
        total_queries = len(queries)
        self.logger.info(f"共加载 {total_queries} 个查询待处理")
        
        # 使用并行处理
        results = await self.process_batch_parallel(
            queries=queries,
            desc="Processing",
            max_workers=max_workers
        )
        
        # 计算总耗时
        end_time = datetime.now()
        duration = end_time - start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        # 打印统计信息
        self.logger.info("="*50)
        self.logger.info("Pipeline运行完成")
        self.logger.info(f"总查询数: {total_queries}")
        self.logger.info(f"最大并行数: {max_workers}")
        self.logger.info(f"成功处理: {len(results)}, 失败数量: {total_queries - len(results)}, 成功率: {(len(results) / total_queries * 100):.2f}%")
        self.logger.info(f"总耗时: {hours} Hours {minutes} Minutes {seconds} Seconds, 平均每个查询耗时: {duration.total_seconds() / total_queries:.2f} Seconds")
        self.logger.info("="*50)
        self.log_api_stats()
    
    async def run_pipeline(self, data_file: str) -> None:
        """
        串行化运行完整的处理流程

        Args:
            data_file: 数据文件路径
        """
        start_time = datetime.now()
        self.logger.info(f"开始处理数据文件: {data_file}")
        
        # 设置data_file
        self.sql_generator.set_data_file(data_file)

        # 准备数据
        queries = self.prepare_queries(data_file)
        total_queries = len(queries)
        self.logger.info(f"共加载 {total_queries} 个查询待处理")
        
        # 使用串行处理
        results = await self.process_batch(
            queries=queries,
            desc="Processing"
        )
        
        # 计算总耗时
        end_time = datetime.now()
        duration = end_time - start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        # 打印统计信息
        self.logger.info("="*50)
        self.logger.info("Pipeline运行完成")
        self.logger.info(f"总查询数: {total_queries}")
        self.logger.info(f"成功处理: {len(results)}, 失败数量: {total_queries - len(results)}, 成功率: {(len(results) / total_queries * 100):.2f}%")
        self.logger.info(f"总耗时: {hours} Hours {minutes} Minutes {seconds} Seconds, 平均每个查询耗时: {duration.total_seconds() / total_queries:.2f} Seconds")
        self.logger.info("="*50)
        self.log_api_stats()
        
    def log_api_stats(self):
        """打印API调用统计信息"""
        stats_file = os.path.join(self.intermediate.pipeline_dir, "api_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                
            # 命令行只输出简要统计
            self.logger.warning(f"LLM API总调用次数: {stats['total_calls']}")
            self.logger.warning(f"总输入tokens: {stats['total_cost']['input_tokens']}; 总输出tokens: {stats['total_cost']['output_tokens']}")
            
            # 详细统计信息写入统计日志
            self.stats_logger.info("="*50)
            self.stats_logger.info("详细API调用统计")
            self.stats_logger.info("="*50)
            
            self.stats_logger.info("按模型统计:")
            for model, model_stats in stats['models'].items():
                self.stats_logger.info(f"{model}:")
                self.stats_logger.info(f"  调用次数: {model_stats['calls']}")
                self.stats_logger.info(f"  输入tokens: {model_stats['input_tokens']}")
                self.stats_logger.info(f"  输出tokens: {model_stats['output_tokens']}")
                self.stats_logger.info(f"  总tokens: {model_stats['total_tokens']}")
            
            self.stats_logger.info("按模块统计:")
            for module, module_stats in stats['modules'].items():
                self.stats_logger.info(f"{module}:")
                self.stats_logger.info(f"  总调用次数: {module_stats['calls']}")
                for model, model_stats in module_stats['models'].items():
                    self.stats_logger.info(f"  {model}:")
                    self.stats_logger.info(f"    调用次数: {model_stats['calls']}")
                    self.stats_logger.info(f"    总tokens: {model_stats['total_tokens']}")
            self.stats_logger.info("="*50)
        
    async def process(self, 
                    query: str, 
                    database_schema: Dict,
                    query_id: str,
                    source: str = "") -> Dict[str, Any]:
        """
        处理单个查询
        
        Args:
            query: 用户查询
            database_schema: 数据库schema
            query_id: 查询ID
            source: 查询来源
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # Add source to database_schema
        database_schema["source"] = source
        
        # 在每个示例开始时添加分隔符
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
        
        # 保存最终的SQL结果
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
        """批量处理数据集中的查询"""
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
        并行化处理数据集中的查询
        
        Args:
            queries: 查询列表
            desc: 进度条描述
            max_workers: 最大并行数，默认为5
            
        Returns:
            List[Dict]: 处理结果列表
        """
        results = []
        completed = 0
        total = len(queries)
        
        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(max_workers)
        
        # 记录开始信息
        self.logger.info(f"开始并行处理 {total} 个查询，最大并行数: {max_workers}")
        
        async def process_with_semaphore(query):
            """
            使用信号量限制并发数
            """
            async with semaphore:
                nonlocal completed
                try:
                    self.logger.info(
                        f"开始处理查询 [{query['query_id']}] "
                        f"({completed + 1}/{total}, 当前并行任务数: {len(asyncio.all_tasks()) - 1})"
                    )
                    
                    result = await self.process(
                        query=query["query"],
                        database_schema=query["database_schema"],
                        query_id=query["query_id"],
                        source=query["source"]
                    )
                    
                    completed += 1
                    self.logger.info(
                        f"完成查询 [{query['query_id']}] "
                        f"({completed}/{total}, 剩余: {total - completed})"
                    )
                    return result
                except Exception as e:
                    completed += 1
                    self.logger.error(
                        f"处理查询 [{query['query_id']}] 时发生错误: {str(e)} "
                        f"({completed}/{total}, 剩余: {total - completed})"
                    )
                    return None
        
        # 创建所有任务
        tasks = [process_with_semaphore(query) for query in queries]
        
        # 使用tqdm显示进度
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
                    self.logger.error(f"任务执行异常: {str(e)}")
                    pbar.update(1)
                    continue
        
        # 记录完成信息
        success_rate = (len(results) / total) * 100
        self.logger.info(
            f"并行处理完成。总数: {total}, 成功: {len(results)}, "
            f"失败: {total - len(results)}, 成功率: {success_rate:.2f}%"
        )
        
        return results 