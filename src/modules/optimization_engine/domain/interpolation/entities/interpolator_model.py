from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..interfaces.base_interpolator import BaseInterpolator


class InterpolatorModel(BaseModel):
    """
    Represents a trained interpolator model and its associated metadata.
    This is a Domain Entity / Aggregate Root.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the trained model.",
    )
    name: str = Field(
        ...,
        description="A human-readable name for the interpolator model (e.g., 'NN-V1').",
    )
    type: str = Field(
        ...,
        description="The type of interpolator (e.g., 'neural_network', 'geodesic').",
    )
    parameters: dict = Field(
        ..., description="The parameters used to initialize the interpolator."
    )
    fitted_interpolator: BaseInterpolator = Field(
        ..., description="The actual fitted interpolator instance."
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Performance metrics (e.g., MSE, R2)."
    )
    description: str = Field(None, description="A brief description of the model.")
    notes: str = Field(None, description="Any additional notes or observations.")
    collection: str = Field(
        None,
        description="A logical grouping for the model (e.g., 'interpolator-family').",
    )
    # You might add timestamps, training duration, etc.

    class Config:
        arbitrary_types_allowed = (
            True  # Allow Pydantic to handle non-Pydantic types like BaseInterpolator
        )
        # Be cautious with this; for production, consider how to serialize/deserialize BaseInterpolator
