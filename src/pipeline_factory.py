from enum import Enum
from typing import Dict
from .core.llm import LLMBase
from .modules.schema_linking.enhanced_linker import EnhancedSchemaLinker
from .modules.sql_generation.gpt_generator import GPTSQLGenerator
from .modules.sql_generation.dc_refiner_generator import DCRefinerSQLGenerator
from .modules.sql_generation.online_synthesis_refiner_generator import OSRefinerSQLGenerator
from .modules.sql_generation.enhanced_generator import EnhancedSQLGenerator
from .modules.sql_generation.direct_dc_refiner_generator import DirectDCRefineSQLGenerator
from .modules.sql_generation.direct_dc_os_refiner_generator import DirectDCOSRefineSQLGenerator
from .modules.post_processing.skip_post_processing import SkipPostProcessor
from .pipeline import ElephantSQLPipeline

class PipelineLevel(Enum):
    """Pipeline level enumeration"""
    BASIC = "basic"                 # Basic pipeline
    INTERMEDIATE = "intermediate"   # Intermediate pipeline
    ADVANCED = "advanced"           # Advanced pipeline

class PipelineFactory:
    """Pipeline factory class, used to create and manage different levels of pipelines"""
    
    def __init__(self, llm: LLMBase, backbone_model: str = "gpt-4o-mini-2024-07-18", temperature: float = 0.0, max_retries: int = 10):
        self.llm = llm
        self.backbone_model = backbone_model
        self.temperature = temperature
        self.max_retries = max_retries
        self._pipelines: Dict[PipelineLevel, ElephantSQLPipeline] = {}
        
    def _create_schema_linker(self):
        """Create a shared schema linker"""
        return EnhancedSchemaLinker(
            self.llm,
            model=self.backbone_model,
            temperature=self.temperature,
            max_tokens=10000,
            max_retries=self.max_retries
        )
        
    def get_pipeline(self, level: PipelineLevel) -> ElephantSQLPipeline:
        """Get a pipeline of a specific level, create it if it doesn't exist"""
        if level not in self._pipelines:
            schema_linker = self._create_schema_linker()
            
            if level == PipelineLevel.BASIC:
                # Basic pipeline: use a simple GPT generator
                self._pipelines[level] = ElephantSQLPipeline(
                    schema_linker=schema_linker,
                    sql_generator=GPTSQLGenerator(
                        self.llm,
                        model=self.backbone_model,
                        temperature=self.temperature,
                        max_tokens=10000,
                        max_retries=self.max_retries
                    ),
                    post_processor=SkipPostProcessor()
                )
                
            elif level == PipelineLevel.INTERMEDIATE:
                # Intermediate pipeline: use DC+Refiner
                self._pipelines[level] = ElephantSQLPipeline(
                    schema_linker=schema_linker,
                    # sql_generator=DCRefinerSQLGenerator(
                    sql_generator=DirectDCRefineSQLGenerator( # New implementation of Divide and Conquer
                        self.llm,
                        model=self.backbone_model,
                        temperature=self.temperature,
                        max_tokens=10000,
                        max_retries=self.max_retries
                    ),
                    post_processor=SkipPostProcessor()
                )
                
            elif level == PipelineLevel.ADVANCED:
                # Advanced pipeline: use DC+OS+Refiner (EnhancedSQLGenerator)
                self._pipelines[level] = ElephantSQLPipeline(
                    schema_linker=schema_linker,
                    # sql_generator=EnhancedSQLGenerator(
                    sql_generator=DirectDCOSRefineSQLGenerator( # New implementation of Divide and Conquer
                        self.llm,
                        model=self.backbone_model,
                        temperature=self.temperature,
                        max_tokens=10000,
                        max_retries=self.max_retries
                    ),
                    post_processor=SkipPostProcessor()
                )
                
        return self._pipelines[level] 