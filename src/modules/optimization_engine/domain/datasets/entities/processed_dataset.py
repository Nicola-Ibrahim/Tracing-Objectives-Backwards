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
    decisions_train: np.typing.NDArray = Field(..., description="Training input features")
    objectives_train: np.typing.NDArray = Field(..., description="Training target values")
    decisions_test: np.typing.NDArray = Field(..., description="Test input features")
    objectives_test: np.typing.NDArray = Field(..., description="Test target values")

    # fitted normalizers (sklearn-like wrappers)
    decisions_normalizer: BaseNormalizer = Field(..., description="Fitted normalizer for decisions")
    objectives_normalizer: BaseNormalizer = Field(..., description="Fitted normalizer for objectives")

    pareto: Pareto = None

    metadata: dict = Field(default_factory=dict, description="Optional metadata")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        decisions_train: np.ndarray,
        objectives_train: np.ndarray,
        decisions_test: np.ndarray,
        objectives_test: np.ndarray,
        decisions_normalizer: Any,
        objectives_normalizer: Any,
        pareto: Pareto = None,
        metadata: dict = None,
    ) -> Self:
        """Coerce arrays and build the entity."""

        return cls(
            name=name,
            decisions_train=decisions_train,
            objectives_train=objectives_train,
            decisions_test=decisions_test,
            objectives_test=objectives_test,
            decisions_normalizer=decisions_normalizer,
            objectives_normalizer=objectives_normalizer,
            pareto=pareto,
            metadata=metadata,
        )
