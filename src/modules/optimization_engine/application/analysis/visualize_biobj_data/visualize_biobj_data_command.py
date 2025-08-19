from pydantic import BaseModel


class VisualizeBiobjDataCommand(BaseModel):
    data_file_name: str = "pareto_data"
