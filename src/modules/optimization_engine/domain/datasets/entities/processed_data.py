from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ....domain.modeling.interfaces.base_normalizer import BaseNormalizer


class ProcessedData(BaseModel):
    """
    Entity representing the processed (normalized, split) version of the data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Normalized splits
    decisions_train: np.typing.NDArray = Field(
        ..., description="Training input features (normalized)"
    )
    objectives_train: np.typing.NDArray = Field(
        ..., description="Training target values (normalized)"
    )
    decisions_test: np.typing.NDArray = Field(
        ..., description="Test input features (normalized)"
    )
    objectives_test: np.typing.NDArray = Field(
        ..., description="Test target values (normalized)"
    )

    # Fitted normalizers
    decisions_normalizer: BaseNormalizer = Field(
        ..., description="Fitted normalizer for decisions"
    )
    objectives_normalizer: BaseNormalizer = Field(
        ..., description="Fitted normalizer for objectives"
    )

    metadata: dict = Field(
        default_factory=dict, description="Optional metadata for the processing"
    )

    @classmethod
    def create(
        cls,
        *,
        decisions_train: np.ndarray,
        objectives_train: np.ndarray,
        decisions_test: np.ndarray,
        objectives_test: np.ndarray,
        decisions_normalizer: BaseNormalizer,
        objectives_normalizer: BaseNormalizer,
        metadata: dict = None,
    ) -> Self:
        return cls(
            decisions_train=decisions_train,
            objectives_train=objectives_train,
            decisions_test=decisions_test,
            objectives_test=objectives_test,
            decisions_normalizer=decisions_normalizer,
            objectives_normalizer=objectives_normalizer,
            metadata=metadata or {},
        )
