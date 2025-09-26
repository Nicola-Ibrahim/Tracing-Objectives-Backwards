from datetime import UTC, datetime
from typing import Self

import numpy as np
from pydantic import BaseModel, Field

from ..value_objects.pareto import Pareto


def _iso_timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()


class GeneratedDataset(BaseModel):
    """Aggregate root capturing optimized data snapshots with metadata."""

    name: str = Field(..., description="Identifier for the generated dataset.")

    X: np.typing.NDArray = Field(..., description="List of input variable arrays.")
    y: np.typing.NDArray = Field(..., description="List of output variable arrays.")

    pareto: Pareto = Field(
        ..., description="Snapshot of the optimized decision/objective data."
    )

    created_at: str = Field(
        default_factory=_iso_timestamp,
        description="ISO 8601 timestamp of dataset creation.",
    )

    @classmethod
    def create(
        cls,
        name: str,
        *,
        X: np.typing.NDArray,
        y: np.typing.NDArray,
        pareto: Pareto,
    ) -> Self:
        return cls(name=name, X=X, y=y, pareto=pareto)

    class Config:
        arbitrary_types_allowed = True
