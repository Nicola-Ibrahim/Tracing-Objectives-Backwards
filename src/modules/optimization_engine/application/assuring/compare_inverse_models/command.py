from pydantic import BaseModel, Field, model_validator

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum


class InverseEstimatorCandidate(BaseModel):
    """Represents a specific inverse estimator candidate (type, optional version, and dataset)."""

    type: EstimatorTypeEnum
    version: int | None = Field(
        None,
        description="Specific integer version number (e.g., 1). If None, latest is used.",
    )
    dataset_name: str | None = Field(
        None,
        description="Dataset identifier associated with this estimator.",
    )


class CompareInverseModelsCommand(BaseModel):
    """Command payload for comparing inverse model candidates."""

    candidates: list[InverseEstimatorCandidate] = Field(
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

    @model_validator(mode="after")
    def _validate_datasets(self) -> "CompareInverseModelsCommand":
        if not self.candidates:
            raise ValueError("At least one candidate must be provided.")
        dataset_names = {c.dataset_name or "dataset" for c in self.candidates}
        estimator_keys = {(c.type.value, c.version) for c in self.candidates}
        if len(dataset_names) > 1:
            if len(estimator_keys) != 1:
                raise ValueError(
                    "Comparing across multiple datasets requires a single estimator type/version."
                )
            if next(iter(estimator_keys))[1] is None:
                raise ValueError(
                    "Comparing across multiple datasets requires a fixed estimator version."
                )
        return self
