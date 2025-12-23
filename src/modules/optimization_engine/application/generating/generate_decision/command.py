from pydantic import BaseModel, Field, model_validator

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...assuring.compare_inverse_models.command import InverseEstimatorCandidate


class GenerateDecisionCommand(BaseModel):
    invser_estimators: list[InverseEstimatorCandidate] = Field(
        ...,
        description="list of inverse model candidates (type + version) to use for generation",
    )
    forward_estimator_type: EstimatorTypeEnum = Field(
        ..., description="Name of the forward model to use for verification"
    )
    target_objective: list[float] = Field(
        ..., description="Target point in objective space (y1, y2, ...)"
    )
    distance_tolerance: float = 0.02
    n_samples: int = 250
    diversity_method: str = "euclidean"
    suggestion_noise_scale: float = 0.05
    validation_enabled: bool = True
    feasibility_enabled: bool = True

    @model_validator(mode="after")
    def _validate_single_dataset(self) -> "GenerateDecisionCommand":
        if not self.invser_estimators:
            raise ValueError("At least one inverse estimator must be provided.")
        dataset_names = {
            c.dataset_name or "dataset" for c in self.invser_estimators
        }
        if len(dataset_names) > 1:
            raise ValueError("Decision generation supports only one dataset at a time.")
        return self
