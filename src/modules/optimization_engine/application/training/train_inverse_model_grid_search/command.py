from typing import Any

from pydantic import BaseModel, Field

from ....domain.modeling.value_objects.estimator_params import (
    EstimatorParams,
    ValidationMetricConfig,
)


class TrainInverseModelGridSearchCommand(BaseModel):
    """Payload for grid-search training of inverse (objectives ➝ decisions) estimators."""

    dataset_name: str = Field(
        ...,
        description="Identifier of the processed dataset to use for training.",
        examples=["dataset"],
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters used to initialize/configure the inverse estimator.",
        examples=[{"type": "rbf", "kernel": "thin_plate_spline", "n_neighbors": 10}],
    )

    estimator_performance_metric_configs: list[ValidationMetricConfig] = Field(
        ...,
        description="Validation metrics to compute during training.",
        examples=[[{"type": "MSE", "params": {}}, {"type": "MAE", "params": {}}]],
    )

    random_state: int = Field(
        ...,
        description="Random seed used across train/test split & estimators.",
        examples=[42],
    )

    cv_splits: int = Field(
        ...,
        ge=2,
        description="Number of cross-validation splits to use during grid search.",
        examples=[5],
    )

    tune_param_name: str = Field(
        ...,
        description="Hyperparameter name to tune.",
        examples=["n_neighbors"],
    )
    tune_param_range: list[Any] = Field(
        ...,
        description="Candidate values for the tuned hyperparameter.",
        min_items=1,
        examples=[[5, 10, 20, 40]],
    )

    learning_curve_steps: int = Field(
        ...,
        description="Number of learning-curve steps for deterministic estimators during grid search.",
        examples=[50],
    )

    epochs: int = Field(
        ...,
        description="Epoch count for probabilistic estimators.",
        examples=[100],
    )

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
