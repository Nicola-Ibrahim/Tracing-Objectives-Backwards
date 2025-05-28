import numpy as np
from pydantic import BaseModel, FieldValidationInfo, field_validator


class ObjectivePreferences(BaseModel):
    """
    Validates and stores user preferences for objective trade-offs.
    Ensures weights are non-negative and sum to 1±epsilon.
    """

    time_weight: float
    energy_weight: float

    @field_validator("time_weight", "energy_weight")
    def validate_individual_weights(cls, v):
        """Ensure each weight is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v

    @field_validator("energy_weight")
    def validate_weight_sum(cls, v: float, info: FieldValidationInfo) -> float:
        """Ensure total weights sum to 1±0.001"""
        if not info.data:  # Handle case where no data exists yet
            return v

        time_weight = info.data.get("time_weight", 0.0)
        if not np.isclose(v + time_weight, 1.0, atol=0.001):
            raise ValueError("Weights must sum to 1.0 ± 0.001")
        return v

    @field_validator("time_weight", "energy_weight")
    def validate_non_negative(cls, v: float) -> float:
        """Ensure weights are non-negative"""
        if v < 0:
            raise ValueError("Weights must be non-negative")
        return v
