from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .dtos import InverseDecisionMapperParams


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
    random_state: Optional[int] = Field(
        None,
        description="Controls the shuffling applied to the data before applying the split. "
        "Pass an int for reproducible output across multiple function calls.",
    )

    base_name: str = Field(
        ...,
        description="The base name for the model file. This will be used to construct the final "
        "file name where the trained model will be saved.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
