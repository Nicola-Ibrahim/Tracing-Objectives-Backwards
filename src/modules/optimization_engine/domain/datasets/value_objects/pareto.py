from typing import Self

import numpy as np
from pydantic import BaseModel, Field


class Pareto(BaseModel):
    """
    Abstraction for multi-objective optimization results, including decision variables,
    objectives, constraints, and constraint violations.

    Attributes:
        X (np.ndarray): Decision variables, shape (n_samples, n_vars) of the final population
        F (np.ndarray): Objective values, shape (n_samples, n_objs) of the final population
        G (np.ndarray): Constraint violations per constraint, shape (n_samples, n_constraints)
        CV (np.ndarray): Aggregated constraint violation (e.g. max, sum), shape (n_samples,)
        history (list): Optional list of intermediate optimization results
    """

    set: np.typing.NDArray = Field(
        ..., description="Decision variables of the Pareto-optimal set"
    )
    front: np.typing.NDArray = Field(
        ..., description="Objective values of the Pareto-optimal front"
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def num_solutions(self) -> int:
        """Returns the number of Pareto-optimal solutions."""
        return self.set.shape[0]

    @classmethod
    def create(cls, set: np.typing.NDArray, front: np.typing.NDArray) -> Self:
        """Factory method to create a Pareto object.

        Args:
            set (np.ndarray): Decision variables of the Pareto-optimal set.
            front (np.ndarray): Objective values of the Pareto-optimal front.

        Returns:
            Pareto: An instance of the Pareto class.
        """
        return cls(set=set, front=front)
