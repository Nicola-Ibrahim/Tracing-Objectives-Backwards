from datetime import UTC, datetime
from typing import Optional, Self

import numpy as np
from pydantic import BaseModel, Field

from ..value_objects.pareto import Pareto
from .processed_data import ProcessedData


def _iso_timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()


class Dataset(BaseModel):
    """
    Aggregate root representing a dataset, which includes the raw optimization results
    (decisions and objectives) and optionally a processed transformation of that data.
    """

    name: str = Field(..., description="Unique identifier for the dataset")

    # Raw Data
    decisions: np.typing.NDArray = Field(..., description="Raw decision variables")
    objectives: np.typing.NDArray = Field(..., description="Raw objective values")
    pareto: Pareto = Field(
        ..., description="Pareto set and front associated with the raw data"
    )

    # Processed part (Optional)
    processed: Optional[ProcessedData] = Field(
        None, description="Processed version of the data (split/normalized)"
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
        pareto: Pareto,
        processed: Optional[ProcessedData] = None,
    ) -> Self:
        return cls(
            name=name,
            decisions=decisions,
            objectives=objectives,
            pareto=pareto,
            processed=processed,
        )

    def add_processed_visuals(self, processed: ProcessedData) -> None:
        """Attaches processed data to the dataset."""
        self.processed = processed
