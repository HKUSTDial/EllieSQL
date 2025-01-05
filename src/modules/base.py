from typing import Any, Dict, Optional, Tuple, Callable
from ..core.intermediate import IntermediateResult
from ..core.utils import TextExtractor
from abc import ABC, abstractmethod
from loguru import logger

class ModuleBase(ABC):
    """所有模块的基类"""
    def __init__(self, name: str, max_retries: int = 5, pipeline_id: str = None):
        self.name = name
        self.intermediate = IntermediateResult(name, pipeline_id)
        self.max_retries = max_retries
        self.extractor = TextExtractor()
        self.logger = None  # 将由pipeline设置
        
    def log_io(self, input_data: Any, output_data: Any):
        """记录模块的输入输出，一级key单独成行"""
        if self.logger:
            self.logger.debug("-"*70)
            self.logger.debug("Module Input:")
            
            # 处理输入数据
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    self.logger.debug(f"  {key}: {value}")
            else:
                self.logger.debug(f"  {input_data}")
            
            self.logger.debug("-"*70)    
            self.logger.debug("\nModule Output:")
            # 处理输出数据
            if isinstance(output_data, dict):
                for key, value in output_data.items():
                    self.logger.debug(f"  {key}: {value}")
            else:
                self.logger.info(f"  {output_data}")
                
            self.logger.info("="*70)
        
    def save_intermediate(self, 
                        input_data: Dict[str, Any],
                        output_data: Dict[str, Any],
                        model_info: Dict[str, Any],
                        query_id: Optional[str] = None) -> str:
        """保存中间结果"""
        return self.intermediate.save_result(
            input_data=input_data,
            output_data=output_data,
            model_info=model_info,
            query_id=query_id
        )
        
    def load_previous(self, 
                     query_id: str,
                     previous_module: str) -> Dict[str, Any]:
        """加载上一个模块的结果"""
        return self.intermediate.load_previous_result(
            query_id=query_id,
            previous_module=previous_module
        )
        
    async def execute_with_retry(self,
                               func: Callable,
                               extract_method: str,
                               error_msg: str,
                               *args,
                               **kwargs) -> Tuple[str, Any]:
        """
        使用重试机制执行操作
        
        Args:
            func: 要执行的异步函数
            extract_method: 使用的提取方法 ('sql' 或 'schema_json')
            error_msg: 失败时的错误信息
            *args, **kwargs: 传递给func的参数
            
        Returns:
            Tuple[str, Any]: (原始输出, 提取的结果)
            
        Raises:
            ValueError: 当重试次数达到上限时抛出
        """
        extractor = (self.extractor.extract_sql if extract_method == 'sql' 
                    else self.extractor.extract_schema_json)
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw_output = await func(*args, **kwargs)
                extracted = extractor(raw_output)
                if extracted is not None:
                    return raw_output, extracted
            except Exception as e:
                last_error = e
                continue
                
        error_message = f"[!] {error_msg} 在{self.max_retries}次尝试后失败。最后一次错误: {str(last_error)}"
        raise ValueError(error_message)