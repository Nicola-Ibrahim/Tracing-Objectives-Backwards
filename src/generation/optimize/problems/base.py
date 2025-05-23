from abc import ABC

import numpy as np
from pymoo.core.problem import Problem

from ...domain.value_objects import Vehicle


class EVProblem(ABC, Problem):
    def __init__(
        self, n_var: int, n_obj: int, n_constr: int, xl: np.ndarray, xu: np.ndarray
    ):
        super().__init__(
            n_var=n_var,  # Number of decision variables
            n_obj=n_obj,  # Number of objectives
            n_constr=n_constr,  # Number of constraints
            xl=xl,  # Lower bounds for decision variables
            xu=xu,  # Upper bounds for decision variables
        )
        self.vehicle: Vehicle = None  # Placeholder for vehicle instance

    def _evaluate(self, X, out, *args, **kwargs): ...
