import pickle
from datetime import datetime
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer, field_validator

from ..interfaces.base_estimator import BaseEstimator
from ..value_objects.loss_history import LossHistory
from ..value_objects.metrics import Metrics


class ModelArtifact(BaseModel):
    """
    Represents a trained model model and its associated metadata.
    This entity encapsulates the actual fitted model instance
    along with essential identifying and descriptive information for a *specific training run*.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this specific training run/model version.",
    )

    parameters: dict[str, Any] = Field(
        ...,
        description="The parameters used to initialize or configure this specific model instance/run.",
    )
    estimator: BaseEstimator = Field(
        ...,
        description="The actual fitted inverse decision mapper instance from this run.",
    )

    trained_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp indicating when this model version was trained.",
    )

    version: int | None = Field(
        default=None,
        description="Automatically assigned sequential version number for the training run",
    )

    metrics: Metrics = Field(description="Metrics fo the the artifact")

    loss_history: LossHistory = Field(
        description="Training loss history for the model."
    )

    # ---- (De)serialization for the estimator field only ----

    @field_serializer("estimator")
    def serialize_model(self, obj: Any) -> bytes:
        """Serializes the object to a byte stream using pickle."""
        return pickle.dumps(obj)

    @field_validator(
        "estimator",
        mode="before",
    )
    @classmethod
    def validate_and_deserialize_object(cls, obj: Any) -> Any:
        """Deserializes the byte stream back into the original object."""
        if isinstance(obj, bytes):
            return pickle.loads(obj)
        return obj

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }  # Ensure datetime is serialized to ISO format
        json_decoders = {
            datetime: datetime.fromisoformat
        }  # Ensure datetime is deserialized from ISO format

    @classmethod
    def from_data(
        cls,
        id: str,
        parameters: dict[str, Any],
        estimator: BaseEstimator,
        metrics: Metrics,
        loss_history: LossHistory,
        trained_at: datetime | None = None,
        version: int | None = None,
    ) -> Self:
        """
        Convenience factory that fills sensible defaults; pass only what you have.
        """

        return cls(
            id=id,
            estimator=estimator,
            parameters=parameters,
            metrics=metrics,
            loss_history=loss_history,
            trained_at=trained_at,
            version=version,
        )

    @classmethod
    def create(
        cls,
        parameters: dict[str, Any],
        estimator: BaseEstimator,
        metrics: Metrics,
        loss_history: LossHistory,
    ) -> Self:
        """
        Convenience factory that fills sensible defaults; pass only what you have.
        """

        return cls(
            estimator=estimator,
            parameters=parameters,
            metrics=metrics,
            loss_history=loss_history,
        )
