from typing import List

from pydantic import BaseModel, Field

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum


class GenerateDecisionCommand(BaseModel):
    inverse_estimator_type: EstimatorTypeEnum = Field(
        ..., description="Name of the interpolator to load"
    )
    forward_estimator_type: EstimatorTypeEnum = Field(
        ..., description="Name of the forward model to use for verification"
    )
    target_objective: List[float] = Field(
        ..., description="Target point in objective space (y1, y2, ...)"
    )
    distance_tolerance: float = 0.02
    n_samples: int = 250
    diversity_method: str = "euclidean"
    suggestion_noise_scale: float = 0.05
    validation_enabled: bool = True
    feasibility_enabled: bool = True
