from pydantic import BaseModel, Field

from ..dtos import InverseDecisionMapperParams, MetricConfig, NormalizerConfig


class CrossValidationConfig(BaseModel):
    n_splits: int = Field(5, description="Number of folds for K-fold cross-validation.")
    shuffle: bool = Field(
        True, description="Whether to shuffle the data before splitting into folds."
    )
    random_state: int = Field(
        None, description="Random state for shuffling, if shuffle is True."
    )


class TrainCvInterpolatorCommand(BaseModel):
    """
    Command for training an interpolator model using K-fold cross-validation.
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

    version_number: int = Field(..., description="The number of training run")

    cross_validation_config: CrossValidationConfig = Field(
        ..., description="Configuration for K-fold cross-validation."
    )
    generate_plots_per_fold: bool = Field(
        False, description="Whether to generate plots for each cross-validation fold."
    )

    class Config:
        arbitrary_types_allowed = True
