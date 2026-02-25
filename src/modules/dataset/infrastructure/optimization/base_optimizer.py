from abc import ABC, abstractmethod
from typing import Any


class BaseOptimizer(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self) -> Any:
        """Run the optimizer and return optimizer-specific run data."""


class ConfigOnlyOptimizer(BaseOptimizer):
    """
    Optimizer that requires only configuration.
    """

    pass


class ProblemAwareOptimizer(BaseOptimizer):
    """
    Optimizer that requires a problem + algorithm in addition to config.
    """

    def __init__(self, problem, algorithm, config: dict[str, Any]):
        super().__init__(config)
        self.problem = problem
        self.algorithm = algorithm
