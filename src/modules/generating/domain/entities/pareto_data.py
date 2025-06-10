from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator


class ParetoDataModel(BaseModel):
    """Pydantic model for storing Pareto front/set data with metadata."""

    pareto_set: np.ndarray
    pareto_front: np.ndarray
    problem_name: str | None = None
    metadata: dict[str, Any] = {}

    @property
    def num_solutions(self) -> int:
        """Number of solutions in the Pareto set/front."""
        return len(self.pareto_set)

    @field_validator("pareto_front", "pareto_set", mode="before")
    def validate_array(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(v)}")
        return v

    @field_validator("pareto_front", mode="after")
    def check_dimensions_match(cls, v, info):
        pareto_set = info.data.get("pareto_set")
        if pareto_set is not None and pareto_set.shape[0] != v.shape[0]:
            raise ValueError(
                "pareto_front and pareto_set must have the same number of rows"
            )
        return v

    class Config:
        arbitrary_types_allowed = True  # ✅ Allow np.ndarray, etc.
