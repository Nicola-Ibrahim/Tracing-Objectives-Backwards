from pydantic import BaseModel, Field

from ...dtos import EstimatorParams


class BaseOODCalibratorParams(BaseModel):
    percentile: float = Field(
        default=97.5,
        description="Percentile used when fitting the OOD calibrator threshold.",
    )
    cov_reg: float = Field(
        default=1e-6,
        description="Covariance regularisation term for the OOD calibrator.",
    )


class BaseConformalCalibratorParams(BaseModel):
    confidence: float = Field(
        default=0.90,
        description="Confidence level for the conformal calibrator.",
    )


class CalibrateDecisionValidationCommand(BaseModel):
    """Parameters required to fit and persist validation calibrators."""

    scope: str = Field(
        ..., description="Logical scope (e.g., estimator type) the calibration targets."
    )
    dataset_name: str = Field(
        default="dataset",
        description="Identifier of the processed dataset to use for calibration.",
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Configurations for the estimator used in the forward models.",
    )
    ood_calibrator_params: BaseOODCalibratorParams = Field(
        default_factory=BaseOODCalibratorParams,
        description="Parameters for the out-of-distribution (OOD) calibrator.",
    )
    conformal_calibrator_params: BaseConformalCalibratorParams = Field(
        default_factory=BaseConformalCalibratorParams,
        description="Parameters for the conformal calibrator.",
    )
