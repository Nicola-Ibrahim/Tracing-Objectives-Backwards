from pydantic import BaseModel, Field

from ...dtos import EstimatorParams, ValidationMetricConfig


class TrainInverseModelCrossValidationCommand(BaseModel):
    """Payload for k-fold inverse (objectives ➝ decisions) training."""

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters used to initialize/configure the inverse estimator.",
    )

    estimator_performance_metric_configs: list[ValidationMetricConfig] = Field(
        [
            ValidationMetricConfig(type="MSE"),
            ValidationMetricConfig(type="MAE"),
        ],
        description="Validation metrics to compute during training.",
    )

    random_state: int = Field(
        42,
        description="Random seed used across train/test split & estimators.",
    )

    tandem_forward_estimator_type: str | None = Field(
        "mdn",
        description="Estimator type string for the latest forward model to use in tandem loss.",
    )

    tandem_weight: float = Field(
        0.0,
        ge=0.0,
        description="Non-negative weight applied to the tandem loss term. 0 disables tandem.",
    )

    cv_splits: int = Field(
        5,
        ge=2,
        description="Number of cross-validation splits.",
    )

    learning_curve_steps: int = Field(
        50, description="Number of learning-curve steps for deterministic estimators."
    )

    epochs: int = Field(100, description="Epoch count for probabilistic estimators.")

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
