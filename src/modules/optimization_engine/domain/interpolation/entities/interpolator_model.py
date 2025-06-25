from datetime import datetime  # Import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

# Assuming this path is correct for BaseInverseDecisionMapper
from ..interfaces.base_inverse_decision_mappers import BaseInverseDecisionMapper


class InterpolatorModel(BaseModel):
    """
    Represents a trained interpolator model and its associated metadata.
    This entity encapsulates the actual fitted interpolator instance
    along with essential identifying and descriptive information for a *specific training run*.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this specific training run/model version.",
    )
    name: str = Field(
        ...,
        description="A human-readable conceptual name for the interpolator type "
        "(e.g., 'f1_vs_f2_PchipMapper'). Multiple versions can share this name.",
    )
    interpolator_type: str = Field(
        ...,
        description="The type or category of interpolator (e.g., 'Pchip', 'Linear ND', 'neural_network').",
    )
    parameters: Any = Field(
        ...,
        description="The parameters used to initialize or configure this specific interpolator instance/run.",
    )
    inverse_decision_mapper: BaseInverseDecisionMapper = Field(
        ...,
        description="The actual fitted inverse decision mapper instance from this run.",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics specific to this training run.",
    )
    trained_at: datetime = (
        Field(  # New field: Timestamp of when this specific model version was trained
            default_factory=datetime.now,
            description="Timestamp indicating when this model version was trained.",
        )
    )
    training_data_identifier: str = Field(
        None,
        description="Identifier for the dataset or data version used to train this model.",
    )
    description: str = Field(
        None, description="A brief description of this specific model version/run."
    )
    notes: str = Field(
        None,
        description="Any additional notes or observations about this model version.",
    )
    collection: str = Field(
        None,
        description="A logical grouping for the model (e.g., '1D_Objective_Mappers', '2D_Decision_Mappers').",
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }  # Ensure datetime is serialized to ISO format
        json_decoders = {
            datetime: datetime.fromisoformat
        }  # Ensure datetime is deserialized from ISO format
