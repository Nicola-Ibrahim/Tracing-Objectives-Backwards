from pymoo.optimize import minimize

from .domain.value_objects import OptimizationResult
from .optimize.algorithms.base import OptimizationAlgorithm
from .optimize.problems.base import Problem


class OptimizationExecutor:
    def __init__(self, problem: Problem, algorithm: OptimizationAlgorithm):
        self.problem = problem
        self.algorithm = algorithm.configure()

    def run(self, generations: int) -> OptimizationResult:
        result = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            termination=("n_gen", generations),
            seed=42,
            verbose=False,
        )
        return OptimizationResult(X=result.X, F=result.F, G=result.G, CV=result.CV)
