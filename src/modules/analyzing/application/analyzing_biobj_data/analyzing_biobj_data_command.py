from pydantic import BaseModel


class AnalyzeBiobjDataCommand(BaseModel):
    results_path: str
    output_path: str = "plots/"
