from pydantic import BaseModel, Field

from ....domain.modeling.value_objects.estimator_params import (
    EstimatorParams,
    ValidationMetricConfig,
)


class TrainInverseModelCommand(BaseModel):
    """Command payload for single-split inverse (objectives ➝ decisions) training."""

    dataset_name: str = Field(
        ...,
        description="Identifier of the processed dataset to use for training.",
        examples=["dataset"],
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters (hyperparameters, configuration) used to initialize/configure "
        "this specific inverse estimator instance for training.",
        examples=[{"type": "mdn"}],
    )

    estimator_performance_metric_configs: list[ValidationMetricConfig] = Field(
        ...,
        description="Configurations for the validation metrics.",
        examples=[[{"type": "MSE", "params": {}}, {"type": "MAE", "params": {}}]],
    )

    random_state: int = Field(
        ...,
        description="Random seed used across train/test split & estimators.",
        examples=[42],
    )

    learning_curve_steps: int = Field(
        ...,
        description="Number of learning-curve steps for deterministic estimators.",
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
