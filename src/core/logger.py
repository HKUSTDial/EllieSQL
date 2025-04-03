import sys
from loguru import logger
from pathlib import Path
from datetime import datetime
import os
from .config import Config

class LoggerManager:
    """Logger manager"""
    
    def __init__(self, pipeline_id: str = None):
        if pipeline_id is None:
            pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Use the configured path
        self.log_dir = Config().logs_dir / "run" / pipeline_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove the default sink
        logger.remove()
        
        # Add console output (warning及以上级别)
        logger.add(
            sys.stderr,
            format="<yellow>{time:YYYY-MM-DD HH:mm:ss}</yellow> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="WARNING"
        )
        
    def get_logger(self, name: str):
        """
        Get the module-specific logger
        
        :param name: Module name
        """
        # Add file output (debug or higher level)
        log_file = self.log_dir / f"{name}.log"
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            filter=lambda record: record["extra"].get("name") == name,
            level="DEBUG"
        )
        
        # Create a context logger with module name
        return logger.bind(name=name) 