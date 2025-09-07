from pydantic import BaseModel


class VisualizeDatasetCommand(BaseModel):
    data_file_name: str = "dataset"
    processed_file_name: str = "dataset"
