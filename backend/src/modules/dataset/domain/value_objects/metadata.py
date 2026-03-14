from datetime import UTC, datetime
from pydantic import BaseModel, Field

def _iso_timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()

class DatasetMetadata(BaseModel):
    """
    Value object containing dataset summary statistics and configuration.
    """
    n_samples: int = Field(default=0, description="Total number of samples")
    n_train: int = Field(default=0, description="Number of training samples")
    n_test: int = Field(default=0, description="Number of testing samples")
    split_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    random_state: int = Field(default=42)
    created_at: str = Field(
        default_factory=_iso_timestamp,
        description="ISO 8601 timestamp of dataset creation.",
    )
