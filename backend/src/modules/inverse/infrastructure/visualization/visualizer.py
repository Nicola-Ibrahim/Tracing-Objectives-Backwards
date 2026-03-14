from typing import Any

from ....dataset.domain.interfaces.base_visualizer import BaseVisualizer
from ....shared.config import ROOT_PATH
from .panels.context_plots import (
    create_combined_context_figure,
)


class PlotlyContextVisualizer(BaseVisualizer):
    """
    Visualizes generation candidates against the background context.
    Saves plots to reports/generation/figures.
    """

    def __init__(self, output_dir: str = "reports/generation/figures"):
        super().__init__(ROOT_PATH / output_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def plot(self, data: dict[str, Any]) -> None:
        """
        Expects:
        - original_objectives: np.ndarray
        - original_decisions: np.ndarray
        - target_objective: np.ndarray
        - candidate_objectives: np.ndarray
        - candidate_decisions: np.ndarray
        """
        # 1. Create Combined Figure
        fig = create_combined_context_figure(
            original_objectives=data["original_objectives"],
            target_objective=data["target_objective"],
            candidate_objectives=data["candidate_objectives"],
            original_decisions=data["original_decisions"],
            candidate_decisions=data["candidate_decisions"],
            vertices_indices=data.get("vertices_indices"),
        )

        # 2. Show
        fig.show()
