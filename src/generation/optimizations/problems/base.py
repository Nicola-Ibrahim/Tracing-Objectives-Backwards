from abc import ABC, abstractmethod

import numpy as np
from pymoo.core.problem import Problem

from .specs import ProblemSpec


class BaseEVProblem(ABC, Problem):
    """Abstract base class for electric vehicle optimization problems."""

    def __init__(self, spec: ProblemSpec):
        """
        Initialize common problem structure using standardized specification.

        Args:
            spec: Complete problem configuration containing:
                - Vehicle parameters
                - Mission requirements
                - Optimization constraints
        """
        self.spec = spec
        self.vehicle = spec.vehicle

        # Initialize pymoo problem through abstract methods
        super().__init__(
            n_var=self._get_var_count(),
            n_obj=self._get_objective_count(),
            n_constr=self._get_constraint_count(),
            xl=self._get_lower_bounds(),
            xu=self._get_upper_bounds(),
        )

    @abstractmethod
    def _get_var_count(self) -> int:
        """Return number of decision variables"""

    @abstractmethod
    def _get_objective_count(self) -> int:
        """Return number of optimization objectives"""

    @abstractmethod
    def _get_constraint_count(self) -> int:
        """Return number of problem constraints"""

    @abstractmethod
    def _get_lower_bounds(self) -> np.ndarray:
        """Return lower bounds for decision variables"""

    @abstractmethod
    def _get_upper_bounds(self) -> np.ndarray:
        """Return upper bounds for decision variables"""
