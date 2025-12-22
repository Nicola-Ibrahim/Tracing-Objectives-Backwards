from pydantic import BaseModel


class VisualizeDatasetCommand(BaseModel):
    dataset_name: str | None = None
    data_file_name: str = "dataset"
    processed_file_name: str = "dataset"
