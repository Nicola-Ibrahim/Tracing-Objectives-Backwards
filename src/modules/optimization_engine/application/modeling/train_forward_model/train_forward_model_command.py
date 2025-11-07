from typing import Any

from pydantic import BaseModel, Field

from ...dtos import EstimatorParams, ValidationMetricConfig


class TrainForwardModelCommand(BaseModel):
    """Command payload for forward (decisions ➝ objectives) training runs."""

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

    cv_splits: int = Field(
        1,
        ge=1,
        description="Number of cross-validation splits. Values greater than 1 trigger "
        "k-fold training; 1 keeps the single train/test workflow.",
    )

    tune_param_name: str | None = Field(
        None,
        description="Hyperparameter name to tune (optional).",
    )
    tune_param_range: list[Any] | None = Field(
        None,
        description="List of candidate values for the tuned hyperparameter.",
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
