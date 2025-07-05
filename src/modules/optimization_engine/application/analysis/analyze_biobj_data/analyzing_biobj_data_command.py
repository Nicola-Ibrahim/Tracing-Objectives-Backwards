from pydantic import BaseModel


class AnalyzeBiobjDataCommand(BaseModel):
    data_file_name: str = "pareto_data"
