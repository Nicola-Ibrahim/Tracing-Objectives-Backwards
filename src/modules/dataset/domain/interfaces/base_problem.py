import numpy as np
from pymoo.core.problem import Problem


class BaseProblem(Problem):
    """
    Base class for optimization problems.
    This class enforces a standard initialization and evaluation method
    for all derived problem classes.
    """

    def __init__(
        self, n_var: int, n_obj: int, n_constr: int, xl: np.ndarray, xu: np.ndarray
    ):
        """Enforce standard problem initialization"""
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Mandatory evaluation method"""
        raise NotImplementedError("Subclasses must implement _evaluate")
