from pydantic import BaseModel, Field
from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum

class ValidateInverseModelCommand(BaseModel):
    """Command payload for validating an inverse model using a forward simulator."""

    inverse_estimator_type: EstimatorTypeEnum = Field(
        ...,
        description="Type of the inverse estimator to validate (e.g., 'mdn', 'cvae')."
    )

    inverse_version_id: str | None = Field(
        None,
        description="Specific version ID for the inverse model. If None, the latest version is used."
    )

    forward_estimator_type: EstimatorTypeEnum = Field(
        ...,
        description="Type of the forward estimator (simulator) to use for validation."
    )   

    forward_version_id: str | None = Field(
        None,
        description="Specific version ID for the forward model. If None, the latest version is used."
    )

    num_samples: int = Field(
        250,
        description="Number of samples to draw from the inverse model for each target."
    )

    random_state: int = Field(
        42,
        description="Random seed for reproducibility."
    )
