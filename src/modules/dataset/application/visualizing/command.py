from pydantic import BaseModel, Field


class VisualizeDatasetCommand(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier to visualize.",
        examples=["dataset"],
    )
