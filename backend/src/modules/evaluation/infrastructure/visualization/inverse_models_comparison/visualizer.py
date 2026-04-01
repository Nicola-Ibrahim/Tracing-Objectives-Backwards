import plotly.graph_objects as go

from .....shared.config import ROOT_PATH
from ....domain.aggregates.diagnostic_report import DiagnosticReport
from ....domain.interfaces.base_visualizer import BaseVisualizer
from .color_utils import get_model_colors
from .panels.accuracy_bias_dispersion_plot import create_accuracy_bias_dispersion_figure
from .panels.accuracy_ecdf_plot import create_accuracy_ecdf_figure
from .panels.calibration_plot import create_calibration_figure
from .panels.metrics_bar_plot import create_metric_bar_figure


class InverseModelsComparisonVisualizer(BaseVisualizer):
    """
    Infrastructure service for visualizing inverse model comparison results.
    Generates and persists high-resolution diagnostic dashboards.
    """

    def __init__(self, output_dir: str = "reports/evaluation/figures"):
        super().__init__(ROOT_PATH / output_dir)

    def plot(self, results: list[DiagnosticReport]) -> None:
        """
        Generates and persists each diagnostic plot based on the data contract.
        """
        # 1. Prepare Data Containers
        calibration_data = {}
        ecdf_data = {}
        bias_data = {}
        dispersion_data = {}
        metric_data = {
            "mace": {},
            "crps": {},
            "diversity": {},
            "sharpness": {},
            "winkler": {},
        }

        model_names = []
        for res in results:
            model_name = f"{res.engine.type} (v{res.engine.version})"
            model_names.append(model_name)

            # a. Objective Space
            ecdf_data[model_name] = {
                "x_sorted": res.objective_space.ecdf_profile.x_values,
                "y_ecdf": res.objective_space.ecdf_profile.cumulative_probabilities,
                "label": model_name,
                "median_val": res.objective_space.median_best_shot,
            }
            # Note: The new domain only provides means for bias/dispersion.
            # We wrap them in a list for compatibility with existing box plots.
            bias_data[model_name] = [res.objective_space.mean_bias]
            dispersion_data[model_name] = [res.objective_space.mean_dispersion]

            # b. Decision Space
            ds = res.decision_space
            if hasattr(ds, "mace"):  # DecisionSpaceDistributionAssessment
                metric_data["mace"][model_name] = ds.mace
                metric_data["crps"][model_name] = ds.mean_crps
                metric_data["diversity"][model_name] = ds.mean_diversity
                metric_data["sharpness"][model_name] = ds.mean_interval_width

                calibration_data[model_name] = {
                    "pit_values": ds.calibration_curve.nominal_coverage,
                    "cdf_y": ds.calibration_curve.empirical_coverage,
                }
            elif hasattr(ds, "mean_coverage_error"):  # DecisionSpaceIntervalAssessment
                metric_data["mace"][model_name] = (
                    ds.mean_coverage_error
                )  # Use error as a proxy
                metric_data["sharpness"][model_name] = ds.mean_interval_width
                metric_data["winkler"][model_name] = ds.mean_winkler_score

                calibration_data[model_name] = {
                    "pit_values": ds.ecdf_profile.x_values,
                    "cdf_y": ds.ecdf_profile.cumulative_probabilities,
                }

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

        # self._persist_figure(
        #     create_error_boxplot_figure(
        #         residuals_data=residuals_data,
        #         color_map=color_map,
        #         model_names=model_names,
        #         title="Residual Distributions: Spread of Best-Shot Errors",
        #         subtitle="Lower is Better",
        #     ),
        #     "accuracy_residual_distributions",
        # )

        # Exploration Metrics
        # self._persist_figure(
        #     create_metric_bar_figure(
        #         metric_values=metric_data["diversity"],
        #         color_map=color_map,
        #         model_names=model_names,
        #         title="Candidate Diversity Score",
        #         subtitle="Higher = More Exploration",
        #     ),
        #     "exploration_diversity",
        # )

        # self._persist_figure(
        #     create_metric_bar_figure(
        #         metric_values=metric_data["sharpness"],
        #         color_map=color_map,
        #         model_names=model_names,
        #         title="Prediction Interval Width (90%)",
        #         subtitle="Lower = More Precise/Sharp",
        #     ),
        #     "sharpness_interval_width",
        # )

    def _persist_figure(self, fig: go.Figure, name: str) -> None:
        """Persists figure to files without timestamps."""
        # Save High-Resolution Static Image (PNG)
        png_path = self.save_path / f"{name}.png"
        fig.write_image(png_path, scale=5)
