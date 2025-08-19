from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import for type checking only to avoid runtime circular imports
    from ....infrastructure.optimizers.result import OptimizationResult


class BaseOptimizer(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self) -> "OptimizationResult":
        """Run the optimizer and return an OptimizationResult.

        The return type is a forward reference to avoid importing the concrete
        `OptimizationResult` at runtime and creating circular dependencies.
        """


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
