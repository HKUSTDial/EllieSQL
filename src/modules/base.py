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
        self._previous_module = None  # 添加前置模块引用
        
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
                        query_id: Optional[str] = None,
                        module_name: Optional[str] = None) -> str:
        """
        保存中间结果
        
        Args:
            input_data: 输入数据
            output_data: 输出数据
            model_info: 模型信息
            query_id: 查询ID
            module_name: 可选的模块名称，如果不指定则使用默认模块名
            
        Returns:
            str: 保存的文件路径
        """
        # 使用指定的模块名或默认模块名
        save_name = module_name if module_name else self.name
        
        return self.intermediate.save_result(
            input_data=input_data,
            output_data=output_data,
            model_info=model_info,
            query_id=query_id,
            module_name=save_name
        )
        
    async def execute_with_retry(self,
                               func: Callable,
                               extract_method: str,
                               error_msg: str,
                               *args,
                               **kwargs) -> Tuple[str, Any]:
        """
        使用重试机制执行操作
        ** NOTE: 该方法已弃用。请使用各模块base类中的专用retry实现。
        
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
        
    def set_previous_module(self, module: 'ModuleBase'):
        """设置前置模块"""
        self._previous_module = module
        
    def load_previous_result(self, query_id: str, module_name: Optional[str] = None) -> Dict:
        """
        加载前置模块的中间结果
        
        Args:
            query_id: 查询ID
            module_name: 可选的模块名称，如果不指定则使用前置模块的默认名称
            
        Returns:
            Dict: 前置模块的处理结果
            
        Raises:
            ValueError: 当没有设置前置模块且未指定module_name时抛出
        """
        if not module_name and not self._previous_module:
            raise ValueError(f"Module {self.name} has no previous module set and no module_name specified")
        
        # 使用指定的模块名或前置模块的默认名称
        load_name = module_name if module_name else self._previous_module.name
        
        return self.intermediate.load_previous_result(
            query_id=query_id,
            previous_module=load_name
        )