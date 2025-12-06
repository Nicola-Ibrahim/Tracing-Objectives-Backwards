from pydantic import BaseModel, Field

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum


class ModelCandidate(BaseModel):
    """Represents a specific model candidate (type and optional version)."""

    type: EstimatorTypeEnum
    version: int | None = Field(
        None,
        description="Specific integer version number (e.g., 1). If None, latest is used.",
    )


class SelectInverseModelCommand(BaseModel):
    """Command payload for selecting the best inverse model by comparing candidates."""

    candidates: list[ModelCandidate] = Field(
        ...,
        description="List of model candidates to compare.",
    )

    forward_estimator_type: EstimatorTypeEnum = Field(
        ...,
        description="Type of the forward estimator (simulator) to use for validation.",
    )

    num_samples: int = Field(
        250,
        description="Number of samples to draw from the inverse model for each target.",
    )

    random_state: int = Field(42, description="Random seed for reproducibility.")
