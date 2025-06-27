from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ....domain.interpolation.entities.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from .dtos import InterpolatorParams


class TrainInterpolatorCommand(BaseModel):
    """
    Pydantic Command to encapsulate all data required for training and saving
    an interpolator model. It specifies the interpolator type, its parameters,
    and metadata for the resulting trained model.
    """

    data_file_name: str = Field(
        ...,
        description="The name of the file containing the training data. "
        "This file should be in a format compatible with the training service (e.g., CSV, JSON).",
    )

    model_conceptual_name: str = Field(
        ...,
        description="A human-readable conceptual name for the interpolator type being trained. "
        "This name is used to group different versions of the same interpolator (e.g., 'f1_vs_f2_PchipMapper').",
    )

    type: InverseDecisionMapperType = Field(
        ...,
        description="The type of interpolator to be trained (e.g., NEURAL_NETWORK, GEODESIC).",
    )
    params: InterpolatorParams = Field(
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

    description: Optional[str] = Field(
        None,
        description="A brief description of this specific training run/model version.",
    )
    notes: Optional[str] = Field(
        None,
        description="Any additional notes or observations about this training run.",
    )
    collection: Optional[str] = Field(
        None,
        description="A logical grouping for the model (e.g., '1D_Objective_Mappers', '2D_Decision_Mappers').",
    )

    # New: Identifier for the training data used in this specific run
    training_data_identifier: Optional[str] = Field(
        None,
        description="An identifier for the dataset used for this training run, if different from data_source_path.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
