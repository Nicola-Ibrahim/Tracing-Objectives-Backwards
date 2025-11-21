from typing import List

from pydantic import BaseModel, Field

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum


class GenerateDecisionCommand(BaseModel):
    estimator_type: EstimatorTypeEnum = Field(
        ..., description="Name of the interpolator to load"
    )
    target_objective: List[float] = Field(
        ..., description="Target point in objective space (y1, y2, ...)"
    )
    distance_tolerance: float = 0.02
    num_suggestions: int = 3
    diversity_method: str = "euclidean"
    suggestion_noise_scale: float = 0.05
    validation_enabled: bool = True
    feasibility_enabled: bool = True
