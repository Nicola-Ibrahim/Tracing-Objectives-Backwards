from pydantic import BaseModel, Field

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum


class SelectInverseModelCommand(BaseModel):
    """Command payload for selecting the best inverse model by comparing candidates."""

    inverse_estimator_types: list[EstimatorTypeEnum] = Field(
        ...,
        description="List of inverse estimator types to compare (e.g., ['mdn', 'cvae']).",
    )

    # We remove inverse_version_id as we likely want to compare the LATEST versions of the specified types.
    # Or we could allow a map of type -> version_id, but for now we keep it simple.

    forward_estimator_type: EstimatorTypeEnum = Field(
        ...,
        description="Type of the forward estimator (simulator) to use for validation.",
    )

    forward_version_id: str | None = Field(
        None,
        description="Specific version ID for the forward model. If None, the latest version is used.",
    )

    num_samples: int = Field(
        250,
        description="Number of samples to draw from the inverse model for each target.",
    )

    random_state: int = Field(42, description="Random seed for reproducibility.")
