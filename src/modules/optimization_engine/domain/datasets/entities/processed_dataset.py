from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict


class ProcessedDataset(BaseModel):
    """
    What we save to .pkl (pickle-friendly).
    We allow arbitrary types so the fitted normalizers can be stored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str

    # normalized splits
    X_train: np.typing.NDArray
    y_train: np.typing.NDArray
    X_test: np.typing.NDArray
    y_test: np.typing.NDArray

    # fitted normalizers (sklearn-like wrappers)
    X_normalizer: Any
    y_normalizer: Any

    # optional original (raw) arrays
    pareto_set: np.typing.NDArray | None = None
    pareto_front: np.typing.NDArray | None = None

    # a little metadata (optional)
    metadata: dict = {}

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
        pareto_set: np.ndarray = None,
        pareto_front: np.ndarray = None,
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
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            metadata=metadata,
        )
