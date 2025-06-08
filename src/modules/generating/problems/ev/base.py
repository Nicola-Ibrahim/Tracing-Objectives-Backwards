import numpy as np

from ..base import BaseProblem
from .config import ProblemConfig
from .vehicle import Vehicle


class EVProblem(BaseProblem):
    """Abstract base class for electric vehicle optimization problems."""

    def __init__(
        self,
        spec: ProblemConfig,
        vehicle: Vehicle,
        n_var: int,
        n_obj: int,
        n_constr: int,
        xl: np.ndarray,
        xu: np.ndarray,
    ):
        """
        Initialize common problem structure using standardized specification.

        Args:
            spec: Complete problem configuration containing:
                - Vehicle parameters
                - Mission requirements
                - Optimization constraints
        """
        self.spec = spec
        self.vehicle = vehicle
        super().__init__(n_var, n_obj, n_constr, xl, xu)
