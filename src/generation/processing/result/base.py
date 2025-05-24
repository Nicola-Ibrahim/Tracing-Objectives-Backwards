from abc import ABC

from ...optimizations.result_handlers import OptimizationResultHandler


class BaseResultProcessor(ABC):
    def __init__(self) -> None: ...
