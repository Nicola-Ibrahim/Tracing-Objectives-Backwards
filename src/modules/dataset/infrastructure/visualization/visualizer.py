from typing import Any

import plotly.graph_objects as go

from ....shared.config import ROOT_PATH
from ...domain.interfaces.base_visualizer import BaseVisualizer
from .panels.pareto_plot import (
    create_pareto_front_figure,
    create_pareto_set_figure,
)
from .panels.raw_density_plot import (
    create_raw_decision_density_figure,
    create_raw_objective_density_figure,
)
from .panels.raw_distributions_plot import (
    create_raw_decision_distributions_figure,
    create_raw_objective_distributions_figure,
)
from .panels.raw_space_plot import (
    create_raw_decision_space_figure,
    create_raw_objective_space_figure,
)
from .panels.three_d_views_plot import (
    create_3d_decision_context_figure,
    create_3d_objective_context_figure,
)


class PlotlyDatasetVisualizer(BaseVisualizer):
    """
    Visualizes bi-objective data using a dict payload.
    Generates and saves multiple figures to the reports/data/figures directory.
    """

    def __init__(self, output_dir: str = "reports/data/figures"):
        super().__init__(ROOT_PATH / output_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def plot(self, data: dict[str, Any]):
        # 1. Unpack Data
        X_raw = data["X_raw"]
        y_raw = data["y_raw"]
        pareto_set = data["pareto_set"]
        pareto_front = data["pareto_front"]

        # 2. Raw Space Plots
        self._persist_figure(
            create_raw_decision_space_figure(pareto_set, X_raw),
            "raw_decision_space",
        )
        self._persist_figure(
            create_raw_objective_space_figure(pareto_front, y_raw),
            "raw_objective_space",
        )

        # 3. Pareto-only Plots
        self._persist_figure(create_pareto_set_figure(pareto_set), "pareto_set")
        self._persist_figure(create_pareto_front_figure(pareto_front), "pareto_front")

        # # 4. Raw Distributions (KDEs)
        # self._persist_figure(
        #     create_raw_decision_distributions_figure(X_raw),
        #     "raw_decision_distribution",
        # )
        # self._persist_figure(
        #     create_raw_objective_distributions_figure(y_raw),
        #     "raw_objective_distribution",
        # )

        # # 5. Raw Density (2D PDFs)
        # self._persist_figure(
        #     create_raw_decision_density_figure(X_raw),
        #     "raw_decision_density",
        # )
        # self._persist_figure(
        #     create_raw_objective_density_figure(y_raw),
        #     "raw_objective_density",
        # )

        # 6. 3D Context Views
        self._persist_figure(
            create_3d_decision_context_figure(X_raw, y_raw), "3d_decision_context"
        )
        self._persist_figure(
            create_3d_objective_context_figure(X_raw, y_raw), "3d_objective_context"
        )

    def _persist_figure(self, fig: go.Figure, name: str) -> None:
        """Persists figure to files."""
        png_path = self.save_path / f"{name}.png"
        try:
            fig.write_image(str(png_path), scale=3)
            # fig.write_html(str(png_path.with_suffix(".html")), include_plotlyjs="cdn")
        except Exception as e:
            print(f"Error saving {name} PNG: {e}")
