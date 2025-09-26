"""Tolerance value object for feasibility checks."""

from pydantic import BaseModel, Field


class Tolerance(BaseModel):
    """Represents a scalar tolerance used by feasibility policies."""

    value: float = Field(..., ge=0, description="Tolerance radius in normalized space.")
