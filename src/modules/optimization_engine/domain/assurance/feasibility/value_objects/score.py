"""Feasibility score value object."""

from pydantic import BaseModel, ConfigDict, field_validator


class Score(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: float

    @field_validator("value")
    def _non_negative(cls, v: float) -> float:  # type: ignore[override]
        if v < 0.0:
            raise ValueError("Score must be non-negative.")
        return v


__all__ = ["Score"]
