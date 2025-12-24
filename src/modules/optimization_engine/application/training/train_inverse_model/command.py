from pydantic import BaseModel, Field

from ....domain.modeling.value_objects.estimator_params import (
    EstimatorParams,
    ValidationMetricConfig,
)


class TrainInverseModelCommand(BaseModel):
    """Command payload for single-split inverse (objectives ➝ decisions) training."""

    dataset_name: str = Field(
        default="dataset",
        description="Identifier of the processed dataset to use for training.",
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters (hyperparameters, configuration) used to initialize/configure "
        "this specific inverse estimator instance for training.",
    )

    estimator_performance_metric_configs: list[ValidationMetricConfig] = Field(
        [
            ValidationMetricConfig(type="MSE"),
            ValidationMetricConfig(type="MAE"),
        ],
        description="Configurations for the validation metrics.",
    )

    random_state: int = Field(
        42,
        description="Random seed used across train/test split & estimators.",
    )

    learning_curve_steps: int = Field(
        50, description="Number of learning-curve steps for deterministic estimators."
    )

    epochs: int = Field(100, description="Epoch count for probabilistic estimators.")

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
