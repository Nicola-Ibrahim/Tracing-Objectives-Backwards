from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class InterpolatorModel(BaseModel):
    name: str = Field(..., description="Model name used for tracking")
    model: Any = Field(..., description="Fitted interpolator object")
    notes: str | None = Field(None, description="Free-form notes about the model")
    model_type: str | None = Field(None, description="Type or class name of the model")
    created_at: str | None = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
