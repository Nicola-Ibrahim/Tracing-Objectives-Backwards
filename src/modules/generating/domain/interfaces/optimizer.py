from abc import ABC, abstractmethod
from typing import Any

from ...adapters.optimizers.result import OptimizationResult
from ...domain.interfaces.problem import BaseProblem
from .algorithm import BaseAlgorithm


class BaseOptimizer(ABC):
    def __init__(
        self,
        problem: BaseProblem,
        algorithm: BaseAlgorithm,
        config: Any,
    ):
        self.problem = problem
        self.algorithm = algorithm
        self.config = config

    @abstractmethod
    def run(self) -> OptimizationResult: ...
