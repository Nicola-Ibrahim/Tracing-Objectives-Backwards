from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.shared.domain.interfaces.base_visualizer import BaseVisualizer
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
            subplot_titles=(
                "<b>Probability Integral Transform (PIT) Calibration Curve</b><br><sup>Comparison of empirical PIT distribution against the theoretical uniform ideal.</sup>",
                "<b>Mean Absolute Calibration Error (MACE)</b><br><sup>Scalar quantification of the deviation from perfect calibration (Lower is Better).</sup>",
                "<b>Continuous Ranked Probability Score (CRPS)</b><br><sup>Comprehensive metric for predictive accuracy and distribution sharpness (Lower is Better).</sup>",
                "<b>Lowest Residual Distribution</b><br><sup>Spread of best-case accuracy across test samples</sup>",
                "<b>Mean Lowest Residual</b><br><sup>Average best-case accuracy (Lower is Better)</sup>",
                "<b>Mean Reliability</b><br><sup>Average median residual: typical prediction quality (Lower is Better)</sup>",
                "<b>Diversity</b><br><sup>Average spread of candidate solutions: higher means more exploration</sup>",
                "<b>Interval Width (90%)</b><br><sup>Width of prediction intervals: lower means more precise</sup>",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        model_names = list(results_map.keys())
        color_map = get_model_colors(model_names)

        # --- Row 1: Calibration Metrics ---
        add_calibration_plot(fig, 1, 1, results_map, color_map)
        # Note: Calibration Residual and CRPS are added in add_metrics_bar_plots

        # --- Row 2: Performance Metrics ---
        add_error_boxplot(fig, 2, 1, results_map, color_map, model_names)
        # Note: Mean Lowest Residual and Mean Reliability are added in add_metrics_bar_plots

        # --- Row 3: Uncertainty Metrics ---
        # Note: Diversity and Sharpness are added in add_metrics_bar_plots

        # --- Add all Bar Metrics ---
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
            width=1800,
            margin=dict(t=100, b=50, l=80, r=350),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
            xaxis1_title="Theoretical Quantiles",
            yaxis1_title="Observed Frequency",
            yaxis2_title="Residual Magnitude",
        )

        # Update subplot title font sizes
        fig.update_annotations(font=dict(size=14))

        fig.add_annotation(
            x=1.02,
            y=0.7,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            text=(
                "<b>How to read the tradeoffs</b><br>"
                "• <b>Accurate but Overconfident</b>:<br>Low mean_lowest_residual, narrow intervals,<br>but high calibration_error."
                "<br><br>"
                "• <b>Honest but Vague</b>:<br>Low mean_residual (hits diagonal),<br>but wide intervals (high interval_width)."
                "<br><br>"
                "• <b>The Winner</b>:<br>Lowest CRPS - best compromise<br>between accuracy and uncertainty."
            ),
            font=dict(size=12, color="#333"),
            bordercolor="lightgray",
            borderwidth=1,
            bgcolor="rgba(255, 255, 255, 0.9)",
            width=300,
        )

        fig.show()
