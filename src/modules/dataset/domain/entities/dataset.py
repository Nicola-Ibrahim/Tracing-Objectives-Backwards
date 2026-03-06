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
    """

    name: str = Field(..., description="Unique identifier for the dataset")

    # Core data
    X: np.ndarray = Field(..., description="Decision variables (X)")
    y: np.ndarray = Field(..., description="Objective values (y)")

    # Optional enrichment
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
            return self.X, self.y
        return self.X[self.train_indices], self.y[self.train_indices]

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the subset of data designated for testing."""
        if len(self.test_indices) == 0:
            return (
                np.array([], dtype=self.X.dtype),
                np.array([], dtype=self.y.dtype),
            )
        return self.X[self.test_indices], self.y[self.test_indices]

    @classmethod
    def create(
        cls,
        name: str,
        *,
        X: np.ndarray,
        y: np.ndarray,
        train_indices: np.ndarray | None = None,
        test_indices: np.ndarray | None = None,
        pareto: Optional[Pareto] = None,
        split_ratio: float = 0.2,
        random_state: int = 42,
    ) -> Self:
        from sklearn.model_selection import train_test_split

        if (train_indices is None or len(train_indices) == 0) and (
            test_indices is None or len(test_indices) == 0
        ):
            indices = np.arange(len(X))
            if split_ratio > 0.0:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=split_ratio,
                    random_state=random_state,
                    shuffle=True,
                )
            else:
                train_indices = indices
                test_indices = np.array([], dtype=int)

        return cls(
            name=name,
            X=X,
            y=y,
            pareto=pareto,
            train_indices=train_indices,
            test_indices=test_indices,
            split_ratio=split_ratio,
            random_state=random_state,
        )
