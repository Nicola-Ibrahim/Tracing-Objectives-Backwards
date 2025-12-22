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
            rows=3,
            cols=3,
            specs=[
                [{}, {"colspan": 2}, None],
                [{}, {}, {}],
                [{}, {}, None],
            ],
            subplot_titles=(
                "<b>Calibration Curve</b><br><sup>How well predicted probabilities match observed frequency</sup>",
                "<b>Re-simulation Error</b><br><sup>Distribution of accuracy across multiple samples</sup>",
                "<b>Best Shot Error</b><br><sup>Minimum error found per target (Lower is Better)</sup>",
                "<b>Calibration Error</b><br><sup>Gap between confidence and frequency (Lower is Better)</sup>",
                "<b>CRPS</b><br><sup>Combined accuracy and uncertainty score (Lower is Better)</sup>",
                "<b>Diversity Score</b><br><sup>Exploration/spread of candidates (Higher is Better)</sup>",
                "<b>Sharpness</b><br><sup>Precision/narrowness of intervals (Lower is Better)</sup>",
            ),
            vertical_spacing=0.15,
        )

        model_names = list(results_map.keys())
        color_map = get_model_colors(model_names)

        # --- 1. Calibration Curve (Row 1, Col 1) ---
        add_calibration_plot(fig, 1, 1, results_map, color_map)

        # --- 2. Re-simulation Error Boxplot (Row 1, Col 2-3) ---
        add_error_boxplot(fig, 1, 2, results_map, color_map, model_names)

        # --- 3. Metrics Bar Charts ---
        add_metrics_bar_plots(fig, results_map, color_map, model_names)

        fig.update_layout(
            title={
                "text": "Inverse Model Comparison Analysis",
                "y": 0.98,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 24},
            },
            template="plotly_white",
            height=1400,
            width=1500,
            margin=dict(t=100, b=50, l=80, r=80),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            xaxis1_title="Predicted Confidence",
            yaxis1_title="Observed Frequency",
            yaxis2_title="Error Magnitude",
        )

        # Update subplot title font sizes
        fig.update_annotations(font=dict(size=14))

        fig.add_annotation(
            x=1.02,
            y=0.98,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            text=(
                "<b>How to read the tradeoffs</b><br>"
                "Accurate but Overconfident: Low best_shot_error, "
                "low sharpness value (very narrow), but high calibration_error."
                "<br><br>"
                "Honest but Vague: Low calibration_error (hits the diagonal), "
                "but high sharpness value (very wide intervals)."
                "<br><br>"
                "The Winner: The model that achieves the lowest CRPS, "
                "as it represents the best compromise between being right and being certain."
            ),
            font=dict(size=12, color="gray"),
            bordercolor="lightgray",
            borderwidth=1,
            bgcolor="white",
        )

        return fig
