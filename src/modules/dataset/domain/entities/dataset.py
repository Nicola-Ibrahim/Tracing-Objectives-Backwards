from typing import Optional, Self

import numpy as np
from pydantic import BaseModel, Field

from ..value_objects.metadata import DatasetMetadata
from ..value_objects.pareto import Pareto


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

    # Metadata Value Object (Mandatory)
    metadata: DatasetMetadata = Field(..., description="Dataset metadata")

    # Indices (still kept as direct fields for performance of split operations)
    train_indices: np.ndarray = Field(..., description="Indices for training set")
    test_indices: np.ndarray = Field(..., description="Indices for testing set")

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
        metadata: DatasetMetadata,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        pareto: Optional[Pareto] = None,
    ) -> Self:
        return cls(
            name=name,
            X=X,
            y=y,
            pareto=pareto,
            metadata=metadata,
            train_indices=train_indices,
            test_indices=test_indices,
        )
