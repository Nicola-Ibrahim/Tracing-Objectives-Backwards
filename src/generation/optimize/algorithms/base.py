from typing import Any, Protocol


class OptimizationAlgorithm(Protocol):
    def configure(self) -> Any: ...
