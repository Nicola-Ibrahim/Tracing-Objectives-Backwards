from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from .estimator import Estimator


class DiagnosticRunMetadata(BaseModel):
    """Metadata for a specific diagnostic evaluation run."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this specific evaluation run.",
    )
    run_number: int | None = Field(
        default=None,
        description="Sequential run number (e.g., 1, 2, 3) assigned by storage.",
    )
    estimator: Estimator
    dataset_name: str
    created_at: datetime = Field(default_factory=datetime.now)
    num_samples: int
    scale_method: str = "sd"
