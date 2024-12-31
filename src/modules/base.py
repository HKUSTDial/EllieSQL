from typing import Any


class ModuleBase:
    """所有模块的基类，提供日志功能"""
    def __init__(self, name: str):
        self.name = name
        
    def log_io(self, input_data: Any, output_data: Any):
        """打印模块的输入输出"""
        print(f"\n{'-'*50}")
        print(f"Module: {self.name}")
        print(f"Input: {input_data}")
        print(f"Output: {output_data}")
        print(f"{'-'*50}\n") 