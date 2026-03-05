from datetime import UTC, datetime
from typing import Optional, Self

import numpy as np
from pydantic import BaseModel, Field

from ..value_objects.pareto import Pareto


def _iso_timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()


class Dataset(BaseModel):
    """
    Aggregate root representing a dataset.

    A Dataset knows only about its data (decisions X, objectives y) and
    optional metadata. Pareto data is an optional enrichment provided by
    data sources that compute it (e.g., optimization solvers). CSV or
    simulation-based sources may have no Pareto data, or compute it
    separately and inject it.
    """

    name: str = Field(..., description="Unique identifier for the dataset")

    # Core data — always present
    decisions: np.typing.NDArray = Field(..., description="Decision variables (X)")
    objectives: np.typing.NDArray = Field(..., description="Objective values (y)")

    # Optional enrichment — only populated when the data source provides it
    pareto: Pareto | None = Field(
        None,
        description="Pareto set and front, if available from the data source",
    )

    # Split metadata
    train_indices: np.ndarray = Field(default_factory=lambda: np.array([], dtype=int))
    test_indices: np.ndarray = Field(default_factory=lambda: np.array([], dtype=int))
    split_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    random_state: int = Field(default=42)

    created_at: str = Field(
        default_factory=_iso_timestamp,
        description="ISO 8601 timestamp of dataset creation.",
    )

    class Config:
        arbitrary_types_allowed = True

    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the subset of data designated for training."""
        if len(self.train_indices) == 0:
            return self.decisions, self.objectives
        return self.decisions[self.train_indices], self.objectives[self.train_indices]

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the subset of data designated for testing."""
        if len(self.test_indices) == 0:
            return (
                np.array([], dtype=self.decisions.dtype),
                np.array([], dtype=self.objectives.dtype),
            )
        return self.decisions[self.test_indices], self.objectives[self.test_indices]

    @classmethod
    def create(
        cls,
        name: str,
        *,
        decisions: np.typing.NDArray,
        objectives: np.typing.NDArray,
        pareto: Optional[Pareto] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        split_ratio: float = 0.2,
        random_state: int = 42,
    ) -> Self:
        return cls(
            name=name,
            decisions=decisions,
            objectives=objectives,
            pareto=pareto,
            train_indices=train_indices
            if train_indices is not None
            else np.array([], dtype=int),
            test_indices=test_indices
            if test_indices is not None
            else np.array([], dtype=int),
            split_ratio=split_ratio,
            random_state=random_state,
        )
