import numpy as np
import plotly.graph_objects as go

from .....shared.config import ROOT_PATH
from ....domain.aggregates.diagnostic_result import DiagnosticResult
from ....domain.interfaces.base_visualizer import BaseVisualizer
from .color_utils import get_model_colors
from .panels.accuracy_bias_dispersion_plot import create_accuracy_bias_dispersion_figure
from .panels.accuracy_ecdf_plot import create_accuracy_ecdf_figure
from .panels.calibration_plot import create_calibration_figure
from .panels.error_boxplot import create_error_boxplot_figure
from .panels.metrics_bar_plot import create_metric_bar_figure


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
        # 1. Prepare Data Containers
        calibration_data = {}
        ecdf_data = {}
        bias_data = {}
        dispersion_data = {}
        residuals_data = {}
        metric_data = {
            "mace": {},
            "crps": {},
            "diversity": {},
            "sharpness": {},
        }

        model_names = []
        for res in results:
            name = f"{res.metadata.estimator.type} (v{res.metadata.estimator.version})"
            model_names.append(name)

            # a. Calibration
            if res.reliability and res.reliability.calibration_curve:
                calibration_data[name] = {
                    "pit_values": res.reliability.calibration_curve.pit_values,
                    "cdf_y": res.reliability.calibration_curve.cdf_y,
                }

            # b. ECDF & Residuals
            if res.accuracy:
                scores = res.accuracy.best_shot_scores
                if scores is None and res.accuracy.discrepancy_scores is not None:
                    scores = np.min(res.accuracy.discrepancy_scores, axis=1)

                if scores is not None:
                    residuals_data[name] = scores
                    x_sorted = np.sort(scores)
                    y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
                    ecdf_data[name] = {
                        "x_sorted": x_sorted,
                        "y_ecdf": y_ecdf,
                        "label": f"{name} ({res.metadata.scale_method}, K={res.metadata.num_samples})",
                        "median_val": np.median(scores),
                    }

                # c. Bias & Dispersion
                bias_data[name] = res.accuracy.systematic_bias
                dispersion_data[name] = res.accuracy.cloud_dispersion

            # d. Scalar Metrics
            if res.reliability:
                metric_data["mace"][name] = res.reliability.calibration_error
                metric_data["crps"][name] = res.reliability.crps
                if res.reliability.summary:
                    metric_data["diversity"][name] = (
                        res.reliability.summary.mean_diversity
                    )
                    metric_data["sharpness"][name] = (
                        res.reliability.summary.mean_interval_width
                    )

        color_map = get_model_colors(model_names)

        # 2. Generate and Persist Figures

        # Reliability Metrics
        self._persist_figure(
            create_calibration_figure(
                calibration_data=calibration_data,
                color_map=color_map,
                title="PIT Calibration Curve: Theoretical vs Observed Quantiles",
                subtitle="",
            ),
            "pit_calibration_curve",
        )

        self._persist_figure(
            create_metric_bar_figure(
                metric_values=metric_data["mace"],
                color_map=color_map,
                model_names=model_names,
                title="Mean Absolute Calibration Error (MACE)",
                subtitle="Lower is Better",
            ),
            "reliability_calibration_error_mace",
        )

        self._persist_figure(
            create_metric_bar_figure(
                metric_values=metric_data["crps"],
                color_map=color_map,
                model_names=model_names,
                title="Continuous Ranked Probability Score (CRPS)",
                subtitle="Lower is Better",
            ),
            "reliability_probabilistic_error_crps",
        )

        # Accuracy Metrics
        self._persist_figure(
            create_accuracy_ecdf_figure(
                ecdf_data=ecdf_data,
                color_map=color_map,
                title="Model Attainment: ECDF of Best-Shot Discrepancy",
                subtitle="",
            ),
            "accuracy_ecdf_best_shot",
        )

        self._persist_figure(
            create_accuracy_bias_dispersion_figure(
                bias_data=bias_data,
                dispersion_data=dispersion_data,
                model_names=model_names,
                title="Model Diagnosis: Systematic Bias vs. Statistical Dispersion",
                subtitle="",
            ),
            "accuracy_bias_dispersion_density",
        )

        self._persist_figure(
            create_error_boxplot_figure(
                residuals_data=residuals_data,
                color_map=color_map,
                model_names=model_names,
                title="Residual Distributions: Spread of Best-Shot Errors",
                subtitle="Lower is Better",
            ),
            "accuracy_residual_distributions",
        )

        # Exploration Metrics
        self._persist_figure(
            create_metric_bar_figure(
                metric_values=metric_data["diversity"],
                color_map=color_map,
                model_names=model_names,
                title="Candidate Diversity Score",
                subtitle="Higher = More Exploration",
            ),
            "exploration_diversity",
        )

        self._persist_figure(
            create_metric_bar_figure(
                metric_values=metric_data["sharpness"],
                color_map=color_map,
                model_names=model_names,
                title="Prediction Interval Width (90%)",
                subtitle="Lower = More Precise/Sharp",
            ),
            "sharpness_interval_width",
        )

    def _persist_figure(self, fig: go.Figure, name: str) -> None:
        """Persists figure to files without timestamps."""
        # Save High-Resolution Static Image (PNG)
        png_path = self.save_path / f"{name}.png"
        fig.write_image(png_path, scale=3)
