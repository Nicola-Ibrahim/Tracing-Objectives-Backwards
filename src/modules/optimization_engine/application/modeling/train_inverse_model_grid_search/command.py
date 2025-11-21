from typing import Any

from pydantic import BaseModel, Field

from ...dtos import EstimatorParams, ValidationMetricConfig


class TrainInverseModelGridSearchCommand(BaseModel):
    """Payload for grid-search training of inverse (objectives ➝ decisions) estimators."""

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

    cv_splits: int = Field(
        5,
        ge=2,
        description="Number of cross-validation splits to use during grid search.",
    )

    tune_param_name: str = Field(
        ...,
        description="Hyperparameter name to tune.",
    )
    tune_param_range: list[Any] = Field(
        ...,
        description="Candidate values for the tuned hyperparameter.",
        min_items=1,
    )

    learning_curve_steps: int = Field(
        50,
        description="Number of learning-curve steps for deterministic estimators during grid search.",
    )

    epochs: int = Field(
        100,
        description="Epoch count for probabilistic estimators.",
    )

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
