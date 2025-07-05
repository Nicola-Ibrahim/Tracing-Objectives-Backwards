from pathlib import Path

from pydantic import BaseModel, Field


class AnalyzeIntrepolatorsPerformanceCommand(BaseModel):
    """
    Command to request fetching and visualizing model performance data.
    Uses Pydantic for robust data validation.
    """

    dir_name: Path = Field(
        ..., description="The root directory containing all trained interpolators."
    )
