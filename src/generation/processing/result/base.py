from abc import ABC

from ...optimizations.result_handlers import OptimizationResult


class BaseResultProcessor(ABC):
    def __init__(self) -> None: ...
