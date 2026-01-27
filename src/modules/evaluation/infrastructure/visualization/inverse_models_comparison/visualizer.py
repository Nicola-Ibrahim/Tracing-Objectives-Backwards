from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .....shared.config import ROOT_PATH
from ....domain.interfaces.base_visualizer import BaseVisualizer
from .color_utils import get_model_colors
from .panels.calibration_plot import add_calibration_plot
from .panels.error_boxplot import add_error_boxplot
from .panels.metrics_bar_plot import add_metric_bar_plot


class InverseModelsComparisonVisualizer(BaseVisualizer):
    """
    Infrastructure service for visualizing inverse model comparison results.
    Generates and persists high-resolution diagnostic dashboards.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self._output_dir = ROOT_PATH / output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, data: dict[str, Any]) -> None:
        """
        Generates and persists each diagnostic plot individually.
        """
        results_map = data["results_map"]
        model_names = list(results_map.keys())
        color_map = get_model_colors(model_names)

        # 1. PIT Calibration Curve
        self._save_pit_curve(results_map, color_map, model_names)

        # 2. Scalar Reliability Metrics (MACE, CRPS)
        self._save_metric_bar(
            results_map,
            color_map,
            model_names,
            ["reliability", "calibration_error"],
            "Mean Absolute Calibration Error (MACE)",
            "calibration_error_mace",
            "Lower is Better",
        )
        self._save_metric_bar(
            results_map,
            color_map,
            model_names,
            ["reliability", "crps"],
            "Continuous Ranked Probability Score (CRPS)",
            "probabilistic_error_crps",
            "Lower is Better",
        )

        # 3. Accuracy Metrics
        self._save_error_boxplot(results_map, color_map, model_names)
        self._save_metric_bar(
            results_map,
            color_map,
            model_names,
            ["accuracy", "summary", "mean_best_shot"],
            "Mean Best-Shot Residual (Standardized)",
            "accuracy_mean_best_shot",
            "Lower is Better",
        )
        self._save_metric_bar(
            results_map,
            color_map,
            model_names,
            ["accuracy", "summary", "median_best_shot"],
            "Mean Reliability (Typical Residual)",
            "reliability_mean_residual",
            "Lower is Better",
        )

        # 4. Uncertainty & Exploration
        self._save_metric_bar(
            results_map,
            color_map,
            model_names,
            ["reliability", "summary", "mean_diversity"],
            "Candidate Diversity Score",
            "exploration_diversity",
            "Higher = More Exploration",
        )
        self._save_metric_bar(
            results_map,
            color_map,
            model_names,
            ["reliability", "summary", "mean_interval_width"],
            "Prediction Interval Width (90%)",
            "sharpness_interval_width",
            "Lower = More Precise/Sharp",
        )

    def _save_pit_curve(self, results_map, color_map, model_names):
        fig = make_subplots(rows=1, cols=1)
        add_calibration_plot(fig, 1, 1, results_map, color_map)
        fig.update_layout(
            title="PIT Calibration Curve: Theoretical vs Observed Quantiles",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Observed Frequency",
            template="plotly_white",
            height=800,
            width=1000,
            showlegend=True,
        )
        self._save_fig(fig, "pit_calibration_curve")

    def _save_error_boxplot(self, results_map, color_map, model_names):
        fig = make_subplots(rows=1, cols=1)
        add_error_boxplot(fig, 1, 1, results_map, color_map, model_names)
        fig.update_layout(
            title="Residual Distributions: Spread of Best-Shot Errors",
            yaxis_title="Standardized Residual Magnitude",
            template="plotly_white",
            height=800,
            width=1200,
            showlegend=True,
        )
        self._save_fig(fig, "residual_distributions")

    def _save_metric_bar(
        self,
        results_map,
        color_map,
        model_names,
        path,
        title,
        filename,
        subtitle,
    ):
        fig = make_subplots(rows=1, cols=1)
        add_metric_bar_plot(
            fig,
            results_map,
            color_map,
            model_names,
            path,
            title,
            row=1,
            col=1,
            show_legend=True,
        )
        fig.update_layout(
            title=f"<b>{title}</b><br><sup>{subtitle}</sup>",
            yaxis_title="Metric Value",
            template="plotly_white",
            height=700,
            width=1200,
            showlegend=True,
        )
        self._save_fig(fig, filename)

    def _save_fig(self, fig: go.Figure, name: str) -> None:
        """Persists figure to files without timestamps."""
        # Save Interactive HTML
        # html_path = self._output_dir / f"{name}.html"
        # fig.write_html(html_path)

        # Save High-Resolution Static Image (PNG)
        png_path = self._output_dir / f"{name}.png"
        fig.write_image(png_path, scale=3)
