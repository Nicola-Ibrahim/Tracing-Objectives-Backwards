from pydantic import BaseModel


class AnalyzeBiobjDataCommand(BaseModel):
    filename: str = "pareto_data"
    plot_folder_name: str = "figures"
