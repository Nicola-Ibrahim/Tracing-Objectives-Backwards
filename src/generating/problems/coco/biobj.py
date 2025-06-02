import cocoex
import numpy as np
from cocoex import Problem as COCOProblem

from ..base import BaseProblem
from .config import ProblemConfig


def get_problem(
    problem_name: str = "bbob-biobj",
    function_indices: int = 1,
    instance_indices: int = 1,
    dimensions: int = 2,
) -> COCOProblem:
    """Initialize the bbob-biobj F1 (Sphere/Sphere) problem."""
    suite = cocoex.Suite(
        f"{problem_name}",  # suite_name
        "",  # suite_instance
        f"year: 2016 dimensions:{dimensions} instance_indices:{instance_indices} function_indices:{function_indices}",  # suite_options
    )
    problem = suite.get_problem(0)
    return problem


class COCOBiObjectiveProblem(BaseProblem):
    """
    Adapter for COCO bi-objective problems to pymoo's Problem interface.

    This class wraps a COCO problem instance and adapts it to work with pymoo's
    optimization framework while maintaining the BaseProblem interface.

    """

    def __init__(self, problem: COCOProblem, spec: ProblemConfig):
        """
        Initialize the BiObjectiveProblem with a COCO problem instance.
        Args:
            problem: A COCO problem instance that implements the callable interface.
            lower_bounds: Lower bounds for the decision variables.
            upper_bounds: Upper bounds for the decision variables.
        """
        self.coco_problem = problem

        super().__init__(
            n_var=spec.n_var,
            n_obj=spec.n_obj,
            n_constr=spec.n_constr,
            xl=np.array([spec.xl] * spec.n_var),
            xu=np.array([spec.xu] * spec.n_var),
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate the COCO problem for multiple solutions"""
        F = np.zeros((X.shape[0], self.n_obj))
        for i, x in enumerate(X):
            result = self.coco_problem(x)
            F[i, :] = result[: self.n_obj]
        out["F"] = F
