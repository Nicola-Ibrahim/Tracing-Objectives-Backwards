from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class ParetoDataModel(BaseModel):
    """Pydantic model for storing Pareto front/set data with metadata."""

    pareto_set: np.ndarray = Field(
        ..., description="The decision variables of the final Pareto set."
    )
    pareto_front: np.ndarray = Field(
        ..., description="The objective values of the final Pareto front."
    )
    historical_solutions: np.ndarray | None = Field(
        None, description="Raw historical decision variables from all generations."
    )
    historical_objectives: np.ndarray | None = Field(
        None, description="Raw historical objective values from all generations."
    )
    problem_name: str | None = Field(
        None, description="An optional name for the problem."
    )
    metadata: dict[str, Any] = Field(
        {}, description="A dictionary for any additional metadata."
    )

    @field_validator(
        "pareto_front",
        "pareto_set",
        "historical_solutions",
        "historical_objectives",
        mode="before",
    )
    @classmethod
    def validate_array(cls, v):
        if v is not None and not isinstance(v, np.ndarray):
            raise TypeError(f"Expected np.ndarray or None, got {type(v)}")
        return v

    @model_validator(mode="after")
    def check_dimensions_match(self):
        if self.pareto_set.shape[0] != self.pareto_front.shape[0]:
            raise ValueError(
                "pareto_front and pareto_set must have the same number of rows"
            )
        if (
            self.historical_solutions is not None
            and self.historical_objectives is not None
        ):
            if (
                self.historical_solutions.shape[0]
                != self.historical_objectives.shape[0]
            ):
                raise ValueError(
                    "historical_solutions and historical_objectives must have the same number of rows if provided"
                )
        return self

    @property
    def num_solutions(self) -> int:
        """Number of solutions in the Pareto set/front."""
        return len(self.pareto_set)

    class Config:
        arbitrary_types_allowed = True
