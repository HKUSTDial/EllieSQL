from typing import Dict, Any, List
from datetime import datetime
import json
import os
from tqdm import tqdm
from src.core.llm import LLMBase
from src.modules.schema_linking.base import SchemaLinkerBase
from src.modules.sql_generation.base import SQLGeneratorBase
from src.modules.post_processing.base import PostProcessorBase
from src.core.intermediate import IntermediateResult
from src.core.logger import LoggerManager
from src.core.schema.manager import SchemaManager

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
        self.stats_logger = self.logger_manager.get_logger("statistics")
        
        self.schema_manager = SchemaManager()
        
    def load_json(self, file_path: str) -> Dict:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
            
    def prepare_queries(self, data_file: str) -> List[Dict]:
        """
        准备查询数据
        
        Args:
            data_file: 数据文件路径
            
        Returns:
            List[Dict]: 准备好的查询列表
        """
        # 加载数据
        merge_dev_demo_data = self.load_json(data_file)
        
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
        
    async def run_pipeline(self, data_file: str):
        """运行完整的处理流程"""
        # 设置data_file
        self.sql_generator.set_data_file(data_file)
        
        self.logger.info(f"开始处理数据文件: {data_file}")
        """
        运行完整的pipeline流程
        
        Args:
            data_file: 数据文件路径
        """
        # 准备数据
        queries = self.prepare_queries(data_file)
        
        # 批量处理查询
        results = await self.process_batch(
            queries=queries,
            desc="Processing queries"
        )
        
        # 打印统计信息
        self.print_stats()
        
    def print_stats(self):
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
        
    async def process_batch(self, 
                         queries: List[Dict],
                         desc: str = "Processing") -> List[Dict]:
        """批量处理查询"""
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