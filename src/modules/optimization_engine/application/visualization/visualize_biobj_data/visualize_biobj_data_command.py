from pydantic import BaseModel, Field

from ...dtos import NormalizerConfig


class VisualizeBiobjDataCommand(BaseModel):
    data_file_name: str = "dataset"
    normalizer_config: NormalizerConfig = Field(
        ...,
        description="Configuration for the normalizer applied to objectives (output data).",
    )
