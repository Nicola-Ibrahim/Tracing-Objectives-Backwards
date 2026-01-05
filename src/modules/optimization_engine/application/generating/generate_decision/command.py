from pydantic import BaseModel, Field, model_validator

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...assuring.compare_inverse_models.command import InverseEstimatorCandidate


class GenerateDecisionCommand(BaseModel):
    inverse_estimators: list[InverseEstimatorCandidate] = Field(
        ...,
        description="list of inverse model candidates (type + version) to use for generation",
        examples=[
            [
                {"type": EstimatorTypeEnum.MDN.value, "version": 1, "dataset_name": "dataset"}
            ]
        ],
    )
    forward_estimator_type: EstimatorTypeEnum = Field(
        ...,
        description="Name of the forward model to use for verification",
        examples=[EstimatorTypeEnum.MDN.value],
    )
    target_objective: list[float] = Field(
        ...,
        description="Target point in objective space (y1, y2, ...)",
        examples=[[0.5, 0.8]],
    )
    distance_tolerance: float = Field(
        ...,
        description="Distance tolerance for selecting candidates.",
        examples=[0.02],
    )
    n_samples: int = Field(
        ...,
        description="Number of samples to draw per inverse estimator.",
        examples=[250],
    )
    diversity_method: str = Field(
        ...,
        description="Diversity method for suggestions.",
        examples=["euclidean"],
    )
    suggestion_noise_scale: float = Field(
        ...,
        description="Noise scale applied to suggestions.",
        examples=[0.05],
    )
    validation_enabled: bool = Field(
        ...,
        description="Enable decision validation.",
        examples=[True],
    )
    feasibility_enabled: bool = Field(
        ...,
        description="Enable feasibility checks.",
        examples=[True],
    )

    @model_validator(mode="after")
    def _validate_single_dataset(self) -> "GenerateDecisionCommand":
        if not self.inverse_estimators:
            raise ValueError("At least one inverse estimator must be provided.")
        dataset_names = {c.dataset_name or "dataset" for c in self.inverse_estimators}
        if len(dataset_names) > 1:
            raise ValueError("Decision generation supports only one dataset at a time.")
        return self
