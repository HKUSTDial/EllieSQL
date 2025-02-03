from enum import Enum
from typing import Dict
from .core.llm import LLMBase
from .modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from .modules.sql_generation.gpt_generator import GPTSQLGenerator
from .modules.sql_generation.dc_refiner_generator import DCRefinerSQLGenerator
from .modules.sql_generation.online_synthesis_refiner_generator import OSRefinerSQLGenerator
from .modules.sql_generation.enhanced_generator import EnhancedSQLGenerator
from .modules.post_processing.skip_post_processing import SkipPostProcessor
from .pipeline import ElephantSQLPipeline

class PipelineLevel(Enum):
    """Pipeline级别枚举"""
    BASIC = "basic"                 # 基础pipeline
    INTERMEDIATE = "intermediate"   # 中级pipeline
    ADVANCED = "advanced"           # 高级pipeline

class PipelineFactory:
    """Pipeline工厂类, 用于创建和管理不同级别的pipeline"""
    
    def __init__(self, llm: LLMBase):
        self.llm = llm
        self._pipelines: Dict[PipelineLevel, ElephantSQLPipeline] = {}
        
    def _create_schema_linker(self):
        """创建共用的schema linker"""
        return EnhancedSchemaLinker(
            self.llm,
            model="gpt-4o-mini-2024-07-18",
            temperature=0.0,
            max_tokens=10000
        )
        
    def get_pipeline(self, level: PipelineLevel) -> ElephantSQLPipeline:
        """获取指定级别的pipeline, 如果不存在则创建"""
        if level not in self._pipelines:
            schema_linker = self._create_schema_linker()
            
            if level == PipelineLevel.BASIC:
                # 基础pipeline：使用简单的GPT生成
                self._pipelines[level] = ElephantSQLPipeline(
                    schema_linker=schema_linker,
                    sql_generator=GPTSQLGenerator(
                        self.llm,
                        model="gpt-4o-mini-2024-07-18",
                        temperature=0.0,
                        max_tokens=10000
                    ),
                    post_processor=SkipPostProcessor()
                )
                
            elif level == PipelineLevel.INTERMEDIATE:
                # 中级pipeline：使用DC+Refiner
                self._pipelines[level] = ElephantSQLPipeline(
                    schema_linker=schema_linker,
                    sql_generator=DCRefinerSQLGenerator(
                        self.llm,
                        model="gpt-4o-mini-2024-07-18",
                        temperature=0.0,
                        max_tokens=10000
                    ),
                    post_processor=SkipPostProcessor()
                )
                
            elif level == PipelineLevel.ADVANCED:
                # 高级pipeline：使用DC+OS+Refiner (EnhancedSQLGenerator)
                self._pipelines[level] = ElephantSQLPipeline(
                    schema_linker=schema_linker,
                    sql_generator=EnhancedSQLGenerator(
                        self.llm,
                        model="gpt-4o-mini-2024-07-18",
                        temperature=0.0,
                        max_tokens=10000
                    ),
                    post_processor=SkipPostProcessor()
                )
                
        return self._pipelines[level] 