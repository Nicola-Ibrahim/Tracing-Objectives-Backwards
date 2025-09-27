from pydantic import BaseModel, Field

from ...dtos import EstimatorParams


class OODCalibratorParams(BaseModel):
    method: str = Field(
        default="mahalanobis",
        description="Method used for the OOD calibrator. Supported: 'mahalanobis', 'knn'.",
    )


class ConformalCalibratorParams(BaseModel):
    method: str = Field(
        default="split_conformal_l2",
        description="Method used for the conformal calibrator. Supported: 'split_conformal_l2'.",
    )
    confidence: float = Field(
        default=0.90,
        description="Confidence level for the conformal calibrator.",
    )


class CalibrateDecisionValidationCommand(BaseModel):
    """Parameters required to fit and persist validation calibrators."""

    dataset_name: str = Field(
        default="dataset",
        description="Identifier of the processed dataset to use for calibration.",
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Configurations for the estimator used in the forward models.",
    )
    ood_calibrator_params: OODCalibratorParams = Field(
        default_factory=OODCalibratorParams,
        description="Parameters for the out-of-distribution (OOD) calibrator.",
    )
    conformal_calibrator_params: ConformalCalibratorParams = Field(
        default_factory=ConformalCalibratorParams,
        description="Parameters for the conformal calibrator.",
    )
