import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer, field_validator

from ..interfaces.base_estimator import BaseEstimator
from ..interfaces.base_normalizer import BaseNormalizer


@dataclass
class TrainingHistory:
    """Epoch-level history for iterative models (optional)."""

    epochs: list[int]
    train_loss: list[float]
    val_loss: list[float]


@dataclass
class LearningCurveRecord:
    """One row in the learning curve (train fraction -> scores)."""

    train_fraction: float
    n_train: int
    train_scores: dict[str, float]
    val_scores: dict[str, float]


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

    X_normalizer: BaseNormalizer = Field(
        ...,
        description="The fitted normalizer for the input/decision space (x_train).",
    )
    y_normalizer: BaseNormalizer = Field(
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

    training_history: TrainingHistory | None = Field(
        None,
        description="Epoch-level training history for iterative models.",
    )
    learning_curve: list[LearningCurveRecord] = Field(
        default_factory=list,
        description="Records from a learning curve analysis showing performance vs. training size.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the model training process.",
    )

    @field_serializer("estimator", "X_normalizer", "y_normalizer")
    def serialize_model_and_normalizers(self, obj: Any) -> bytes:
        """Serializes the object to a byte stream using pickle."""
        return pickle.dumps(obj)

    @field_validator(
        "estimator",
        "y_normalizer",
        "X_normalizer",
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
        estimator: BaseEstimator,
        y_normalizer: BaseNormalizer,
        X_normalizer: BaseNormalizer,
    ) -> Self:
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
            estimator=estimator,
            y_normalizer=y_normalizer,
            X_normalizer=X_normalizer,
        )
