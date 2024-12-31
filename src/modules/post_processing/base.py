from ..base import ModuleBase

class PostProcessorBase(ModuleBase):
    def __init__(self, name: str = "PostProcessor"):
        super().__init__(name) 