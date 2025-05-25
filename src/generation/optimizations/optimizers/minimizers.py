from pymoo.optimize import minimize

from ..algorithms.base import BaseOptimizationAlgorithm
from ..problems.base import BaseProblem
from ..result_handlers import OptimizationResult
from .optim_config import MinimizerConfig


class Minimizer:
    def __init__(
        self,
        problem: BaseProblem,
        algorithm: BaseOptimizationAlgorithm,
        config: MinimizerConfig,
    ):
        self.problem = problem
        self.algorithm = algorithm
        self.config = config

    def run(self) -> OptimizationResult:
        result = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            **self.config.model_dump(),
        )
        return OptimizationResult(
            X=result.X, F=result.F, G=result.G, CV=result.CV, history=result.history
        )
