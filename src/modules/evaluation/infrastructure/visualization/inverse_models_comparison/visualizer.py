from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .....shared.config import ROOT_PATH
from ....domain.aggregates.diagnostic_result import DiagnosticResult
from ....domain.interfaces.base_visualizer import BaseVisualizer
from .color_utils import get_model_colors
from .panels.accuracy_bias_dispersion_plot import add_accuracy_bias_dispersion_plot
from .panels.accuracy_ecdf_plot import add_accuracy_ecdf_plot
from .panels.calibration_plot import add_calibration_plot
from .panels.error_boxplot import add_error_boxplot
from .panels.metrics_bar_plot import add_metric_bar_plot


class InverseModelsComparisonVisualizer(BaseVisualizer):
    """
    Infrastructure service for visualizing inverse model comparison results.
    Generates and persists high-resolution diagnostic dashboards.
    """

    def __init__(self, output_dir: str = "reports/evaluation/figures"):
        super().__init__(ROOT_PATH / output_dir)

    def plot(self, results: list[DiagnosticResult]) -> None:
        """
        Generates and persists each diagnostic plot individually based on the data contract.
        """
        # Derive display names and result map
        results_map_dict = {}
        model_names = []
        for res in results:
            name = f"{res.metadata.estimator.type} (v{res.metadata.estimator.version})"
            model_names.append(name)
            results_map_dict[name] = res.model_dump()

        color_map = get_model_colors(model_names)

        # 1. PIT Calibration Curve
        pit_fig = self._create_pit_curve_figure(results_map_dict, color_map)
        self._persist_figure(pit_fig, "pit_calibration_curve")

        # 2. Scalar Reliability Metrics
        mace_fig = self._create_metric_bar_figure(
            results_map_dict,
            color_map,
            model_names,
            path=["reliability", "calibration_error"],
            title="Mean Absolute Calibration Error (MACE)",
            subtitle="Lower is Better",
        )
        self._persist_figure(mace_fig, "reliability_calibration_error_mace")

        crps_fig = self._create_metric_bar_figure(
            results_map_dict,
            color_map,
            model_names,
            path=["reliability", "crps"],
            title="Continuous Ranked Probability Score (CRPS)",
            subtitle="Lower is Better",
        )
        self._persist_figure(crps_fig, "reliability_probabilistic_error_crps")

        # 3. Accuracy Metrics
        ecdf_fig = self._create_accuracy_ecdf_figure(results_map_dict, color_map)
        self._persist_figure(ecdf_fig, "accuracy_ecdf_best_shot")

        bias_disp_fig = self._create_bias_dispersion_diagnosis_figure(
            results_map_dict, model_names
        )
        self._persist_figure(bias_disp_fig, "accuracy_bias_dispersion_density")

        boxplot_fig = self._create_error_boxplot_figure(
            results_map_dict, color_map, model_names
        )
        self._persist_figure(boxplot_fig, "accuracy_residual_distributions")

        # 4. Uncertainty & Exploration
        diversity_fig = self._create_metric_bar_figure(
            results_map_dict,
            color_map,
            model_names,
            path=["reliability", "summary", "mean_diversity"],
            title="Candidate Diversity Score",
            subtitle="Higher = More Exploration",
        )
        self._persist_figure(diversity_fig, "exploration_diversity")

        sharpness_fig = self._create_metric_bar_figure(
            results_map_dict,
            color_map,
            model_names,
            path=["reliability", "summary", "mean_interval_width"],
            title="Prediction Interval Width (90%)",
            subtitle="Lower = More Precise/Sharp",
        )
        self._persist_figure(sharpness_fig, "sharpness_interval_width")

    def _create_pit_curve_figure(self, results_map, color_map) -> go.Figure:
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
        return fig

    def _create_accuracy_ecdf_figure(self, results_map, color_map) -> go.Figure:
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
        return fig

    def _create_bias_dispersion_diagnosis_figure(
        self, results_map, model_names
    ) -> go.Figure:
        fig = go.Figure()
        add_accuracy_bias_dispersion_plot(
            fig=fig, results_map=results_map, model_names=model_names
        )
        fig.update_layout(
            title="<b>Model Diagnosis: Systematic Bias vs. Statistical Dispersion</b>",
            height=800,
            width=1200,
        )
        return fig

    def _create_error_boxplot_figure(
        self, results_map, color_map, model_names
    ) -> go.Figure:
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
        return fig

    def _create_metric_bar_figure(
        self,
        results_map: dict[str, Any],
        color_map: dict[str, Any],
        model_names: list[str],
        path: list[str],
        title: str,
        subtitle: str,
    ) -> go.Figure:
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
        return fig

    def _persist_figure(self, fig: go.Figure, name: str) -> None:
        """Persists figure to files without timestamps."""
        # Save High-Resolution Static Image (PNG)
        png_path = self.save_path / f"{name}.png"
        fig.write_image(png_path, scale=3)
