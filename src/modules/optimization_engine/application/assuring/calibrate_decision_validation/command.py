from pydantic import BaseModel, Field

from ...dtos import EstimatorParams


class OODValidatorParams(BaseModel):
    method: str = Field(
        default="mahalanobis",
        description="Method used for the OOD validator. Supported: 'mahalanobis'.",
    )
    percentile: float = Field(
        default=97.5,
        ge=0,
        le=100,
        description="Percentile used to set the Mahalanobis inlier threshold.",
    )
    cov_reg: float = Field(
        default=1e-6,
        ge=0,
        description="Diagonal covariance regularisation term.",
    )


class ConformalValidatorParams(BaseModel):
    method: str = Field(
        default="split_conformal_l2",
        description="Method used for the conformal validator. Supported: 'split_conformal_l2'.",
    )
    confidence: float = Field(
        default=0.90,
        description="Confidence level for the conformal calibrator.",
    )


class CalibrateDecisionValidationCommand(BaseModel):
    """Parameters required to fit and persist validation validators."""

    dataset_name: str = Field(
        default="dataset",
        description="Identifier of the processed dataset to use for calibration.",
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Configurations for the estimator used in the forward models.",
    )
    ood_validator_params: OODValidatorParams = Field(
        default_factory=OODValidatorParams,
        alias="ood_calibrator_params",
        description="Parameters for the out-of-distribution (OOD) validator.",
    )
    conformal_validator_params: ConformalValidatorParams = Field(
        default_factory=ConformalValidatorParams,
        alias="conformal_calibrator_params",
        description="Parameters for the conformal validator.",
    )


# Backward-compatible names
OODCalibratorParams = OODValidatorParams
ConformalCalibratorParams = ConformalValidatorParams
