from pydantic import BaseModel, Field

from ...dtos import EstimatorParams, ValidationMetricConfig


class TrainForwardModelCommand(BaseModel):
    """Payload for standard forward (decisions ➝ objectives) training."""

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters used to initialize/configure the forward estimator.",
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

    learning_curve_steps: int = Field(
        50, description="Number of learning-curve steps for deterministic estimators."
    )

    epochs: int = Field(
        100, description="Epoch count for probabilistic estimators."
    )

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
