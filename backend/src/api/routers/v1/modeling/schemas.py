from typing import Any, List, Optional

from pydantic import BaseModel, Field

from src.modules.modeling.domain.enums.transform_type import TransformTypeEnum


class TransformationStepSchema(BaseModel):
    """Schema for a single transformation step."""

    type: TransformTypeEnum = Field(..., description="Type of transformation to apply.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the transformation."
    )
    columns: Optional[List[int]] = Field(
        None, description="Optional indices of specific columns to transform."
    )


class TransformationChainSchema(BaseModel):
    """Schema for an ordered sequence of transformations."""

    steps: List[TransformationStepSchema] = Field(
        ..., description="Ordered list of transformation steps."
    )


class TransformationPreviewRequest(BaseModel):
    """Request schema for data transformation preview."""

    dataset_name: str = Field(..., description="Name of the dataset to preview.")
    split: str = Field("train", description="Dataset split to use (train, test, all).")
    sampling_limit: int = Field(
        2000, description="Maximum number of points to sample for preview."
    )
    chain: Optional[List[TransformationStepSchema]] = Field(
        None, description="Sequence of transformations to apply (legacy support)."
    )
    x_chain: List[TransformationStepSchema] = Field(
        default_factory=list, description="Transformation chain for the X space."
    )
    y_chain: List[TransformationStepSchema] = Field(
        default_factory=list, description="Transformation chain for the y space."
    )


class DataPreviewPoints(BaseModel):
    """Holds X and y points for original or transformed data."""

    X: List[List[float]]
    y: List[List[float]]


class TransformationPreviewResponse(BaseModel):
    """Response schema for data transformation preview."""

    original: DataPreviewPoints
    transformed: DataPreviewPoints
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Calculated statistics before/after."
    )


class ParameterDefinition(BaseModel):
    """Schema for a single parameter definition."""

    name: str
    type: str
    required: bool
    default: Any | None = None
    options: list[Any] | None = None
    description: str | None = None


class TransformerSchema(BaseModel):
    """Schema for an entry in the transformer registry."""

    type: TransformTypeEnum
    name: str
    parameters: List[ParameterDefinition]


class TransformerRegistryResponse(BaseModel):
    """Response schema for listing available transformers."""

    transformers: List[TransformerSchema]
