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

    created_at: str = Field(
        default_factory=_iso_timestamp,
        description="ISO 8601 timestamp of dataset creation.",
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(
        cls,
        name: str,
        *,
        decisions: np.typing.NDArray,
        objectives: np.typing.NDArray,
        pareto: Optional[Pareto] = None,
    ) -> Self:
        return cls(
            name=name,
            decisions=decisions,
            objectives=objectives,
            pareto=pareto,
        )
