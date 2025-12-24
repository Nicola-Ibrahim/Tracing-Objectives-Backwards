from pydantic import BaseModel, Field

from ....domain.modeling.value_objects.estimator_params import EstimatorParams


class OODValidatorParams(BaseModel):
    method: str = Field(
        ...,
        description="Method used for the OOD validator. Supported: 'mahalanobis'.",
        examples=["mahalanobis"],
    )
    percentile: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentile used to set the Mahalanobis inlier threshold.",
        examples=[97.5],
    )
    cov_reg: float = Field(
        ...,
        ge=0,
        description="Diagonal covariance regularisation term.",
        examples=[1e-6],
    )


class ConformalValidatorParams(BaseModel):
    method: str = Field(
        ...,
        description="Method used for the conformal validator. Supported: 'split_conformal_l2'.",
        examples=["split_conformal_l2"],
    )
    confidence: float = Field(
        ...,
        description="Confidence level for the conformal calibrator.",
        examples=[0.9],
    )


class CalibrateDecisionValidationCommand(BaseModel):
    """Parameters required to fit and persist validation validators."""

    dataset_name: str = Field(
        ...,
        description="Identifier of the processed dataset to use for calibration.",
        examples=["dataset"],
    )
    estimator_params: EstimatorParams = Field(
        ...,
        description="Estimator configuration used for calibration.",
        examples=[{"type": "mdn"}],
    )

    ood_validator_params: OODValidatorParams = Field(
        ...,
        alias="ood_calibrator_params",
        description="Parameters for the out-of-distribution (OOD) validator.",
    )
    conformal_validator_params: ConformalValidatorParams = Field(
        ...,
        alias="conformal_calibrator_params",
        description="Parameters for the conformal validator.",
    )
