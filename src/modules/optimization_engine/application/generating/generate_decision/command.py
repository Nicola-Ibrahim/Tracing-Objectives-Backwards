from pydantic import BaseModel, Field

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...assuring.compare_inverse_models.command import ModelCandidate


class GenerateDecisionCommand(BaseModel):
    dataset_name: str = Field(
        default="dataset",
        description="Identifier of the processed dataset to use for decision generation.",
    )
    inverse_candidates: list[ModelCandidate] = Field(
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
