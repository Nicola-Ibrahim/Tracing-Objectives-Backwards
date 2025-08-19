from pydantic import BaseModel, Field

from ..dtos import (
    InverseDecisionMapperParams,
    ModelPerformanceMetricConfig,
    NormalizerConfig,
)


class TrainCvModelCommand(BaseModel):
    """
    Command for training an interpolator model using K-fold cross-validation.
    """

    inverse_decision_mappers_params: InverseDecisionMapperParams = Field(
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
    model_performance_metric_config: ModelPerformanceMetricConfig = Field(
        ModelPerformanceMetricConfig(type="MSE"),
        description="Configuration for the validation metric.",
    )

    version_number: int = Field(..., description="The number of training run")

    n_splits: int = Field(5, description="Number of folds for K-fold cross-validation.")
    shuffle: bool = Field(
        True, description="Whether to shuffle the data before splitting into folds."
    )
    random_state: int = Field(
        None, description="Random state for shuffling, if shuffle is True."
    )

    class Config:
        arbitrary_types_allowed = True
