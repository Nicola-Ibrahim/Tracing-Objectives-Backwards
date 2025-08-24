from pydantic import BaseModel, Field

from ..dtos import (
    MlMapperParams,
    NormalizerConfig,
    ValidationMetricConfig,
)


class TrainModelCommand(BaseModel):
    """
    Pydantic Command to encapsulate all data required for training and saving
    an interpolator model. It specifies the interpolator type, its parameters,
    and metadata for the resulting trained model.
    """

    ml_mapper_params: MlMapperParams = Field(
        ...,
        description="Parameters (hyperparameters, configuration) used to initialize/configure "
        "this specific interpolator instance for training.",
    )

    objectives_normalizer_config: NormalizerConfig = Field(
        None,
        description="Configuration for the normalizer applied to objectives (output data).",
    )
    decisions_normalizer_config: NormalizerConfig = Field(
        None,
        description="Configuration for the normalizer applied to decisions (input data).",
    )
    model_performance_metric_configs: list[ValidationMetricConfig] = Field(
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

    cv_splits: int | None = Field(
        None,
        gt=1,
        description="Number of cross-validation splits. If specified, a full cross-validation workflow is executed. Otherwise, a single train-test split is used.",
    )

    class Config:
        arbitrary_types_allowed = True
