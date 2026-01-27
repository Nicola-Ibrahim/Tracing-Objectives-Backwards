from typing import Literal

from pydantic import BaseModel, Field

from .....modeling.domain.enums.estimator_type import EstimatorTypeEnum


class InverseEstimatorCandidate(BaseModel):
    """Represents a specific inverse estimator candidate (type and optional version)."""

    type: EstimatorTypeEnum = Field(
        ...,
        examples=[EstimatorTypeEnum.MDN.value],
    )
    version: int | None = Field(
        ...,
        description="Specific integer version number (e.g., 1). If None, latest is used.",
        examples=[1],
    )


class DiagnoseInverseModelsCommand(BaseModel):
    """
    Command for the full evaluation suite including
    Objective-Space Accuracy and Decision-Space Reliability.
    Supports comparing multiple inverse model candidates.
    """

    dataset_name: str = Field(..., examples=["cocoex_f5"])

    inverse_estimator_candidates: list[InverseEstimatorCandidate] = Field(
        ...,
        description="List of model candidates to compare.",
        examples=[[{"type": EstimatorTypeEnum.MDN.value, "version": 1}]],
    )

    forward_estimator_type: EstimatorTypeEnum = Field(
        ..., examples=[EstimatorTypeEnum.COCO]
    )

    num_samples: int = Field(default=200, description="K candidates per target")
    random_state: int = 42

    scale_method: Literal["sd", "mad", "iqr"] = Field(
        default="sd", description="sd | mad | iqr"
    )
    bias_threshold: float = 0.5
    dispersion_threshold: float = 0.5
