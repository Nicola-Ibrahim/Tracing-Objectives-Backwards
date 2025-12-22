from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....domain.common.interfaces.base_visualizer import BaseVisualizer
from .color_utils import get_model_colors
from .panels.calibration_plot import add_calibration_plot
from .panels.error_boxplot import add_error_boxplot
from .panels.metrics_bar_plot import add_metrics_bar_plots


class InverseComparisonVisualizer(BaseVisualizer):
    """
    Infrastructure service for visualizing inverse model comparison results.
    """

    def plot(self, data: dict[str, Any]) -> go.Figure:
        """
        Generates comparison plots for multiple models in a SINGLE figure.
        Row 1: Calibration Curve (Left), Re-simulation Error Boxplot (Right, Spanned)
        Row 2: Best Shot Error, Reliability Error, Diversity Score (Bar Charts)

        Expected data keys:
        - results_map: dict[str, dict] {model_name: {metrics, raw_errors, calibration}}
        """
        results_map = data["results_map"]

        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{}, {"colspan": 2}, None],
                [{}, {}, {}],
            ],
            subplot_titles=(
                "Calibration Curve (Quantitative)",
                "Re-simulation Error Boxplot",
                "Best Shot Error (Lower is Better)",
                "Calibration Error (Lower is Better)",
                "Diversity Score (Higher is Better)",
            ),
            vertical_spacing=0.15,
        )

        model_names = list(results_map.keys())
        color_map = get_model_colors(model_names)

        # --- 1. Calibration Curve (Row 1, Col 1) ---
        add_calibration_plot(fig, 1, 1, results_map, color_map)

        # --- 2. Re-simulation Error Boxplot (Row 1, Col 2-3) ---
        add_error_boxplot(fig, 1, 2, results_map, color_map, model_names)

        # --- 3. Metrics Bar Charts (Row 2) ---
        add_metrics_bar_plots(fig, results_map, color_map, model_names)

        fig.update_layout(
            title_text="Inverse Model Comparison",
            template="plotly_white",
            height=900,
            width=1400,
            xaxis1_title="Predicted Confidence",
            yaxis1_title="Observed Frequency",
            yaxis2_title="Error",
            yaxis3_title="Error",
            yaxis4_title="Error",
            yaxis5_title="Score",
        )

        return fig
