from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

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

    parameters: dict[Any, Any] = Field(
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
    trained_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp indicating when this model version was trained.",
    )

    version_number: int = Field(..., description="The number of training run")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }  # Ensure datetime is serialized to ISO format
        json_decoders = {
            datetime: datetime.fromisoformat
        }  # Ensure datetime is deserialized from ISO format

    def to_save_format(self) -> dict[str, Any]:
        """
        Converts the InterpolatorModel entity into a serializable dictionary
        format suitable for saving to storage.
        """
        # Use model_dump, but hide this implementation detail from the repository.
        # Pydantic's dump handles datetime serialization to ISO format.
        return self.model_dump(exclude={"inverse_decision_mapper"})

    @classmethod
    def from_saved_format(
        cls, saved_data: dict[str, Any], loaded_mapper: BaseInverseDecisionMapper
    ):
        """
        Constructs an InterpolatorModel entity from saved metadata and a loaded mapper.
        This method will be used by the repository's load() method.
        """
        # Note: Pydantic's constructor handles automatic deserialization of
        # ISO-formatted datetime strings if the field is typed as datetime.
        return cls(
            inverse_decision_mapper=loaded_mapper,
            # We can directly pass the loaded dictionary to the constructor
            **saved_data,
        )
