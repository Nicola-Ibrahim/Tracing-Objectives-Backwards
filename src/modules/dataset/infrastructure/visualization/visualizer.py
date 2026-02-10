from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from ....shared.config import ROOT_PATH
from ...domain.interfaces.base_visualizer import BaseVisualizer
from .panels.normalized_density_plot import (
    create_normalized_decision_density_figure,
    create_normalized_objective_density_figure,
)
from .panels.normalized_distributions_plot import (
    create_normalized_decision_pdf_figure,
    create_normalized_objective_pdf_figure,
)
from .panels.normalized_space_plot import (
    create_normalized_decision_space_figure,
    create_normalized_objective_space_figure,
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

    def plot(self, data: dict[str, Any]):
        if not isinstance(data, dict):
            raise TypeError("Visualizer expects a dict prepared by the handler.")

        # Drop any accidental objects (defensive)
        for k in list(data.keys()):
            if "normalizer" in k.lower():
                data.pop(k, None)

        # 1. Raw Spaces
        self._persist_figure(
            create_raw_decision_space_figure(data), "raw_decision_space"
        )
        self._persist_figure(
            create_raw_objective_space_figure(data), "raw_objective_space"
        )

        # # 2. Raw Distributions (KDEs)
        # self._persist_figure(
        #     create_raw_decision_distributions_figure(data), "raw_decision_distribution"
        # )
        # self._persist_figure(
        #     create_raw_objective_distributions_figure(data),
        #     "raw_objective_distribution",
        # )

        # # 3. Raw Density (2D PDFs)
        # self._persist_figure(
        #     create_raw_decision_density_figure(data), "raw_decision_density"
        # )
        # self._persist_figure(
        #     create_raw_objective_density_figure(data), "raw_objective_density"
        # )

        # 4. Normalized Spaces
        self._persist_figure(
            create_normalized_decision_space_figure(data), "normalized_decision_space"
        )
        self._persist_figure(
            create_normalized_objective_space_figure(data), "normalized_objective_space"
        )

        # # 5. Normalized Distributions
        # self._persist_figure(
        #     create_normalized_decision_pdf_figure(data),
        #     "normalized_decision_distribution",
        # )
        # self._persist_figure(
        #     create_normalized_objective_pdf_figure(data),
        #     "normalized_objective_distribution",
        # )

        # # 6. Normalized Density
        # self._persist_figure(
        #     create_normalized_decision_density_figure(data),
        #     "normalized_decision_density",
        # )
        # self._persist_figure(
        #     create_normalized_objective_density_figure(data),
        #     "normalized_objective_density",
        # )

        # 7. 3D Context Views
        self._persist_figure(
            create_3d_decision_context_figure(data), "3d_decision_context"
        )
        self._persist_figure(
            create_3d_objective_context_figure(data), "3d_objective_context"
        )

    def _persist_figure(self, fig: go.Figure, name: str) -> None:
        """Persists figure to files."""
        png_path = self.save_path / f"{name}.png"
        try:
            # fig.write_image(str(png_path), scale=3)
            fig.write_html(str(png_path.with_suffix(".html")), include_plotlyjs="cdn")
        except Exception as e:
            print(f"Error saving {name} PNG: {e}")
