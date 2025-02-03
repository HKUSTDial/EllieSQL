import sys
from loguru import logger
from pathlib import Path
from datetime import datetime
import os
from .config import Config

class LoggerManager:
    """日志管理器"""
    
    def __init__(self, pipeline_id: str = None):
        if pipeline_id is None:
            pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # 使用配置的路径
        self.log_dir = Config().logs_dir / pipeline_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 移除默认的sink
        logger.remove()
        
        # 添加控制台输出（warning及以上级别）
        logger.add(
            sys.stderr,
            format="<yellow>{time:YYYY-MM-DD HH:mm:ss}</yellow> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="WARNING"
        )
        
    def get_logger(self, name: str):
        """
        获取模块专属的logger
        
        Args:
            name: 模块名称
        """
        # 添加文件输出（debug及以上级别）
        log_file = self.log_dir / f"{name}.log"
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            filter=lambda record: record["extra"].get("name") == name,
            level="DEBUG"
        )
        
        # 创建带有模块名称的上下文logger
        return logger.bind(name=name) 