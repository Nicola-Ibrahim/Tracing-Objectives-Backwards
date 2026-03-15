from typing import Self

import numpy as np
from pydantic import BaseModel, Field


class Pareto(BaseModel):
    """
    Value object representing the Pareto-optimal frontier of a dataset.

    Contains the decision variables (set) and objective values (front)
    of all Pareto-optimal solutions.
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
            set: Decision variables of the Pareto-optimal solutions.
            front: Objective values of the Pareto-optimal solutions.

        Returns:
            Pareto instance.
        """
        return cls(set=set, front=front)
