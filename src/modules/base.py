from typing import Any, Dict, Optional, Tuple, Callable
from ..core.intermediate import IntermediateResult
from ..core.utils import TextExtractor
from abc import ABC, abstractmethod
from loguru import logger

class ModuleBase(ABC):


    """Base class for all modules"""


    def __init__(self, name: str, max_retries: int = 5, pipeline_id: str = None):
        self.name = name
        self.intermediate = IntermediateResult(name, pipeline_id)
        self.max_retries = max_retries
        self.extractor = TextExtractor()
        self.logger = None  # Set by the pipeline
        self._previous_module = None  # Add a reference to the previous module
        self.data_file = None  # Add the data_file attribute
        
    def set_data_file(self, data_file: str):
        """Set the data file path"""
        self.data_file = data_file
    
    def log_io(self, input_data: Any, output_data: Any):
        """Record the input and output of the module, with each first-level key on a new line"""
        if self.logger:
            self.logger.debug("-"*70)
            self.logger.debug("Module Input:")
            
            # Process the input data
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    self.logger.debug(f"  {key}: {value}")
            else:
                self.logger.debug(f"  {input_data}")
            
            self.logger.debug("-"*70)    
            self.logger.debug("\nModule Output:")
            
            # Process the output data
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
        Save the intermediate result
        
        :param input_data: Input data
        :param output_data: Output data
        :param model_info: Model information
        :param query_id: Query ID
        :param module_name: Optional module name, if not specified, use the default module name
            
        :return: The path of the saved file
        """
        # Use the specified module name or the default module name
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
        Use the retry mechanism to execute the operation
        ** NOTE: This method has been deprecated! Please use the dedicated retry implementation in the module base class.
        
        :param func: The asynchronous function to execute
        :param extract_method: The extraction method to use ('sql' or 'schema_json')
        :param error_msg: The error message to display when the operation fails
        :param *args, **kwargs: The arguments to pass to func
            
        :return: Tuple[str, Any]: (raw output, extracted result)
            
        :raises:
            ValueError: When the number of retries reaches the maximum
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
                
        error_message = f"[!] {error_msg} failed after {self.max_retries} attempts. Last error: {str(last_error)}"
        raise ValueError(error_message)
        
    def set_previous_module(self, module: 'ModuleBase'):
        """Set the previous module"""
        self._previous_module = module
        
    def load_previous_result(self, query_id: str, module_name: Optional[str] = None) -> Dict:
        """
        Load the intermediate result of the previous module
        
        :param query_id: Query ID
        :param module_name: Optional module name, if not specified, use the default module name
            
        :return: The result of the previous module
            
        :raises:
            ValueError: When there is no previous module set and no module_name specified
        """
        if not module_name and not self._previous_module:
            raise ValueError(f"Module {self.name} has no previous module set and no module_name specified")
        
        # Use the specified module name or the default module name of the previous module
        load_name = module_name if module_name else self._previous_module.name
        
        return self.intermediate.load_previous_result(
            query_id=query_id,
            previous_module=load_name
        )