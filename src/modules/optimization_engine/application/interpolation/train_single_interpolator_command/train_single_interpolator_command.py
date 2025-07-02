from pydantic import BaseModel, Field

from ..dtos import InverseDecisionMapperParams, MetricConfig, NormalizerConfig


class TrainSingleInterpolatorCommand(BaseModel):
    """
    Pydantic Command to encapsulate all data required for training and saving
    an interpolator model. It specifies the interpolator type, its parameters,
    and metadata for the resulting trained model.
    """

    params: InverseDecisionMapperParams = Field(
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
    validation_metric_config: MetricConfig = Field(
        MetricConfig(type="MSE"),
        description="Configuration for the validation metric.",
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

    version_number: int = Field(..., description="The number of training run")

    class Config:
        arbitrary_types_allowed = True
