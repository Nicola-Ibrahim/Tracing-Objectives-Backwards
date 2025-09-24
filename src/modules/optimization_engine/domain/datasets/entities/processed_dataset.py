from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ....domain.datasets.value_objects.pareto import Pareto
from ....domain.modeling.interfaces.base_normalizer import (
    BaseNormalizer,
)


class ProcessedDataset(BaseModel):
    """
    What we save to .pkl (pickle-friendly).
    We allow arbitrary types so the fitted normalizers can be stored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique name/ID of the dataset")

    # normalized splits
    X_train: np.typing.NDArray = Field(..., description="Training input features")
    y_train: np.typing.NDArray = Field(..., description="Training target values")
    X_test: np.typing.NDArray = Field(..., description="Test input features")
    y_test: np.typing.NDArray = Field(..., description="Test target values")

    # fitted normalizers (sklearn-like wrappers)
    X_normalizer: BaseNormalizer = Field(..., description="Fitted normalizer for X")
    y_normalizer: BaseNormalizer = Field(..., description="Fitted normalizer for y")

    pareto: Pareto = None

    metadata: dict = Field(default_factory=dict, description="Optional metadata")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_normalizer: Any,
        y_normalizer: Any,
        pareto: Pareto = None,
        metadata: dict = None,
    ) -> Self:
        """Coerce arrays and build the entity."""

        return cls(
            name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            pareto=pareto,
            metadata=metadata,
        )
