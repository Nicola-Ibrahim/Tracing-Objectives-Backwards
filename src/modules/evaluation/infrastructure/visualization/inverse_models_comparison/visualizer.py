from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .....shared.config import ROOT_PATH
from ....domain.interfaces.base_visualizer import BaseVisualizer
from .color_utils import get_model_colors
from .panels.accuracy_ecdf_plot import add_accuracy_ecdf_plot
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
        self._save_pit_curve(
            results_map=results_map, color_map=color_map, model_names=model_names
        )

        # 2. Scalar Reliability Metrics (MACE, CRPS)
        self._save_metric_bar(
            results_map=results_map,
            color_map=color_map,
            model_names=model_names,
            path=["reliability", "calibration_error"],
            title="Mean Absolute Calibration Error (MACE)",
            filename="calibration_error_mace",
            subtitle="Lower is Better",
        )
        self._save_metric_bar(
            results_map=results_map,
            color_map=color_map,
            model_names=model_names,
            path=["reliability", "crps"],
            title="Continuous Ranked Probability Score (CRPS)",
            filename="probabilistic_error_crps",
            subtitle="Lower is Better",
        )

        # 3. Accuracy Metrics
        self._save_accuracy_ecdf(results_map, color_map, model_names)
        self._save_error_boxplot(results_map, color_map, model_names)
        self._save_metric_bar(
            results_map=results_map,
            color_map=color_map,
            model_names=model_names,
            path=["accuracy", "summary", "mean_best_shot"],
            title="Mean Best-Shot Residual (Standardized)",
            filename="accuracy_mean_best_shot",
            subtitle="Lower is Better",
        )
        self._save_metric_bar(
            results_map=results_map,
            color_map=color_map,
            model_names=model_names,
            path=["accuracy", "summary", "median_best_shot"],
            title="Mean Reliability (Typical Residual)",
            filename="reliability_mean_residual",
            subtitle="Lower is Better",
        )

        # 4. Uncertainty & Exploration
        self._save_metric_bar(
            results_map=results_map,
            color_map=color_map,
            model_names=model_names,
            path=["reliability", "summary", "mean_diversity"],
            title="Candidate Diversity Score",
            filename="exploration_diversity",
            subtitle="Higher = More Exploration",
        )
        self._save_metric_bar(
            results_map=results_map,
            color_map=color_map,
            model_names=model_names,
            path=["reliability", "summary", "mean_interval_width"],
            title="Prediction Interval Width (90%)",
            filename="sharpness_interval_width",
            subtitle="Lower = More Precise/Sharp",
        )

    def _save_pit_curve(self, results_map, color_map, model_names):
        fig = make_subplots(rows=1, cols=1)
        add_calibration_plot(fig, 1, 1, results_map, color_map)
        fig.update_layout(
            title="<b>PIT Calibration Curve: Theoretical vs Observed Quantiles</b>",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Observed Frequency",
            template="plotly_white",
            height=800,
            width=1000,
            showlegend=True,
        )
        self._save_fig(fig, "pit_calibration_curve")

    def _save_accuracy_ecdf(self, results_map, color_map, model_names):
        fig = make_subplots(rows=1, cols=1)
        add_accuracy_ecdf_plot(
            fig=fig, row=1, col=1, results_map=results_map, color_map=color_map
        )
        fig.update_layout(
            title="<b>Model Attainment: ECDF of Best-Shot Discrepancy</b>",
            xaxis_title="Best-shot discrepancy (min over K)",
            yaxis_title="Fraction of targets",
            yaxis_range=[0, 1.05],
            template="plotly_white",
            height=800,
            width=1000,
            showlegend=True,
        )
        self._save_fig(fig, "accuracy_ecdf_best_shot")

    def _save_error_boxplot(self, results_map, color_map, model_names):
        fig = make_subplots(rows=1, cols=1)
        add_error_boxplot(
            fig=fig,
            row=1,
            col=1,
            results_map=results_map,
            color_map=color_map,
            model_names=model_names,
        )
        fig.update_layout(
            title="<b>Residual Distributions: Spread of Best-Shot Errors</b>",
            yaxis_title="Standardized Residual Magnitude",
            template="plotly_white",
            height=800,
            width=1200,
            showlegend=True,
        )
        self._save_fig(fig, "residual_distributions")

    def _save_metric_bar(
        self,
        results_map: dict[str, Any],
        color_map: dict[str, Any],
        model_names: list[str],
        path: list[str],
        title: str,
        filename: str,
        subtitle: str,
    ):
        fig = make_subplots(rows=1, cols=1)
        add_metric_bar_plot(
            fig=fig,
            results_map=results_map,
            color_map=color_map,
            model_names=model_names,
            metric_path=path,
            metric_name=title,
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
