from typing import Any, Literal

from pydantic import BaseModel, Field

from .dtos import InverseDecisionMapperParams


class NormalizerConfig(BaseModel):
    """
    Configuration for a normalizer.
    """

    type: Literal[
        "MinMaxScaler",
        "HypercubeNormalizer",
        "StandardNormalizer",
        "UnitVectorNormalizer",
        "LogNormalizer",
    ] = Field(..., description="The type of the normalizer to use.")
    params: dict[str, Any] = Field(
        {}, description="Parameters specific to the normalizer type."
    )


class MetricConfig(BaseModel):
    """
    Configuration for a validation metric.
    """

    type: Literal["MSE"] = Field(..., description="The type of the metric to use.")
    params: dict[str, Any] = Field(
        {}, description="Parameters specific to the metric type."
    )


class TrainInterpolatorCommand(BaseModel):
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

    test_size: float = Field(
        0.2,
        description="The proportion of the dataset to include in the test split for validation.",
        ge=0.0,
        le=1.0,
    )
    random_state: int | None = Field(
        None,
        description="Controls the shuffling applied to the data before applying the split. "
        "Pass an int for reproducible output across multiple function calls.",
    )

    version_number: int = Field(..., description="The number of training run")

    should_generate_plots: bool = Field(
        True, description="Enables the generation and saving of visualization plots."
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
        MetricConfig(type="MSE"),  # Default to MSE if not explicitly provided
        description="Configuration for the validation metric.",
    )

    class Config:
        arbitrary_types_allowed = True
