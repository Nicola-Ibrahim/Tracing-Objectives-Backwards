from typing import Any

from pydantic import BaseModel, Field

from ...dtos import (
    EstimatorParams,
    NormalizerConfig,
    ValidationMetricConfig,
)


class TrainModelCommand(BaseModel):
    """
    Pydantic Command to encapsulate all data required for training and saving
    an interpolator model. It specifies the interpolator type, its parameters,
    and metadata for the resulting trained model.
    """

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters (hyperparameters, configuration) used to initialize/configure "
        "this specific interpolator instance for training.",
    )

    normalizer_config: NormalizerConfig = Field(
        None,
        description="Configuration for the normalizer applied to objectives (output data).",
    )

    estimator_performance_metric_configs: list[ValidationMetricConfig] = Field(
        [
            ValidationMetricConfig(type="MSE"),
            ValidationMetricConfig(type="MAE"),
        ],
        description="Configurations for the validation metrics.",
    )

    test_size: float = Field(
        0.2,
        description="The proportion of the dataset to include in the test split for validation.",
        ge=0.0,
        le=1.0,
    )
    random_state: int = Field(
        42,
        description="Controls the shuffling applied to the data before applying the split. "
        "Pass an int for reproducible output across multiple function calls.",
    )

    cv_splits: int = Field(
        1,
        gt=1,
        description="Number of cross-validation splits. If specified, a full cross-validation workflow is executed. Otherwise, a single train-test split is used.",
    )

    tune_param_name: str | None = Field(
        None,
        description="Name of the hyperparameter to tune (e.g., 'C' for SVM, 'n_estimators' for RandomForest).",
    )
    tune_param_range: list[Any] | None = Field(
        None,
        description="A list of values to test for the hyperparameter. Required if tune_param_name is set.",
    )

    class Config:
        arbitrary_types_allowed = True
