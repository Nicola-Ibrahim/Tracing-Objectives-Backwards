from abc import ABC, abstractmethod

import numpy as np
from pymoo.core.problem import Problem


class BaseProblem(Problem, ABC):
    @abstractmethod
    def __init__(
        self, n_var: int, n_obj: int, n_constr: int, xl: np.ndarray, xu: np.ndarray
    ):
        """Enforce standard problem initialization"""
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    @abstractmethod
    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Mandatory evaluation method"""
        super()._evaluate(X, out, *args, **kwargs)
