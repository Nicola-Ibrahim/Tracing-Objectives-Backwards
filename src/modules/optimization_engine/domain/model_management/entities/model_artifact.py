import pickle
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer, field_validator

from ..interfaces.base_ml_mapper import BaseMlMapper
from ..interfaces.base_normalizer import BaseNormalizer


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
    ml_mapper: BaseMlMapper = Field(
        ...,
        description="The actual fitted inverse decision mapper instance from this run.",
    )

    objectives_normalizer: BaseNormalizer = Field(
        ...,
        description="The fitted normalizer for the input/decision space (x_train).",
    )
    decisions_normalizer: BaseNormalizer = Field(
        ...,
        description="The fitted normalizer for the output/objective space (y_train).",
    )

    train_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics for a single training run (e.g., from a train-test split).",
    )
    test_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics for the test set (if applicable).",
    )
    cv_scores: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Aggregated performance metrics (mean and std) from cross-validation. Only present for CV runs.",
    )

    trained_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp indicating when this model version was trained.",
    )

    version: int | None = Field(
        None,
        description="Automatically assigned sequential version number for the training run",
    )

    @field_serializer("ml_mapper", "objectives_normalizer", "decisions_normalizer")
    def serialize_model_and_normalizers(self, obj: Any) -> bytes:
        """Serializes the object to a byte stream using pickle."""
        return pickle.dumps(obj)

    @field_validator(
        "ml_mapper",
        "objectives_normalizer",
        "decisions_normalizer",
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
    def create(
        cls,
        id: str,
        parameters: dict[str, Any],
        train_scores: dict[str, float],
        cv_scores: dict[str, list[float]],
        trained_at: datetime,
        version: int | None,
        ml_mapper: BaseMlMapper,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
    ) -> "ModelArtifact":
        """
        Factory method to create a new ModelArtifact instance.
        """
        return cls(
            id=id,
            parameters=parameters,
            train_scores=train_scores,
            cv_scores=cv_scores,
            trained_at=trained_at,
            version=version,
            ml_mapper=ml_mapper,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
        )
