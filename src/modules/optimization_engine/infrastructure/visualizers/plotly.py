from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d

from ...domain.analyzing.interfaces.base_visualizer import BaseParetoVisualizer


class PlotlyParetoVisualizer(BaseParetoVisualizer):
    """Enhanced dashboard for visualizing Pareto set and front with dedicated legends."""

    def __init__(self, save_path: Path | None = None):
        super().__init__(save_path)
        self._f1_rel_data: dict | None = None
        self._x1_x2_interp_data: dict | None = None
        self._seen_interp_methods = set()
        self._interp_colors = {
            "Pchip": "#E64A19",  # Deep orange
            "Cubic Spline": "#1976D2",  # Blue
            "Linear": "#43A047",  # Green
            "Quadratic": "#7B1FA2",  # Purple
        }

    def plot(self, f1_rel_data: dict, x1_x2_interp_data: dict) -> None:
        """
        Generate an interactive dashboard with enhanced visualization and dedicated legends.
        """
        self._f1_rel_data = f1_rel_data
        self._x1_x2_interp_data = x1_x2_interp_data

        # Safely extract original data for initial plots
        pareto_set_orig = np.hstack(
            [
                f1_rel_data["x1_orig"].reshape(-1, 1),
                f1_rel_data["x2_orig"].reshape(-1, 1),
            ]
        )
        pareto_front_orig = np.hstack(
            [
                f1_rel_data["f1_orig"].reshape(-1, 1),
                f1_rel_data["f2_orig"].reshape(-1, 1),
            ]
        )

        # Setup subplot layout with improved organization
        fig = make_subplots(
            rows=3,
            cols=3,
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "parcoords"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [
                    {"type": "scatter"},
                    {"type": "scatter"},
                    {"type": "scatter"},
                ],  # Three separate plots for f1 relationships
            ],
            subplot_titles=[
                "Decision Space ($x_1$ vs $x_2$)",
                "Objective Space ($f_1$ vs $f_2$)",
                "Parallel Coordinates",
                "Normalized Decision Space",
                "Normalized Objective Space",
                "$x_1$ vs $x_2$ (Interpolations)",
                "$f_1$ vs $f_2$ (Interpolations)",
                "$f_1$ vs $x_1$ (Interpolations)",
                "$f_1$ vs $x_2$ (Interpolations)",
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
            column_widths=[0.3, 0.3, 0.4],
        )

        # Update overall layout with dedicated legend space
        fig.update_layout(
            title=dict(
                text="Pareto Optimization Analysis Dashboard",
                x=0.5,
                font=dict(size=24, color="#2c3e50"),
            ),
            height=1600,
            width=1600,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.12,
                xanchor="center",
                x=0.5,
                title=dict(text="Interpolation Methods", font=dict(size=14)),
            ),
            margin=dict(t=100, b=100, l=50, r=50),
            font=dict(family="Arial", size=12, color="#2c3e50"),
        )

        # Add plots using prepared data
        self._add_decision_objective_spaces(fig, pareto_set_orig, pareto_front_orig)
        self._add_normalized_spaces(fig)
        self._add_parallel_coordinates(fig)
        self._add_x1_x2_interpolation(fig)
        self._add_f1_relationships(fig)  # Now with three separate plots

        # Save and Show
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            fig.write_image(
                file=self.save_path / "enhanced_pareto_dashboard.png",
                width=1600,
                height=1600,
                scale=2,
                engine="kaleido",
            )
        fig.show()

    def _set_axis_limits(self, fig, row, col, x_data, y_data, padding=0.05):
        """Helper to set axis limits with some padding."""
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        x_range = x_max - x_min
        y_range = y_max - y_min

        fig.update_xaxes(
            range=[x_min - padding * x_range, x_max + padding * x_range],
            row=row,
            col=col,
        )
        fig.update_yaxes(
            range=[y_min - padding * y_range, y_max + padding * y_range],
            row=row,
            col=col,
        )

    def _add_description(self, fig, row, col, text):
        """Add description text annotation below a subplot using paper coordinates."""
        # Calculate positions based on grid layout
        row_positions = {1: 0.92, 2: 0.62, 3: 0.32}
        col_positions = {1: 0.15, 2: 0.5, 3: 0.85}

        fig.add_annotation(
            text=f"<i>{text}</i>",
            x=col_positions.get(col, 0.5),
            y=row_positions.get(row, 0),
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=11, color="#7f8c8d"),
            align="center",
        )

    def _add_decision_objective_spaces(
        self, fig: go.Figure, pareto_set_orig: np.ndarray, pareto_front_orig: np.ndarray
    ):
        """Visualize Pareto set and front in original space."""
        # Decision space (Row 1, Col 1)
        fig.add_trace(
            go.Scatter(
                x=pareto_set_orig[:, 0],
                y=pareto_set_orig[:, 1],
                mode="markers",
                marker=dict(
                    size=7,
                    opacity=0.8,
                    color="#3498db",  # Blue
                    line=dict(width=1, color="#2c3e50"),
                ),
                name="Pareto Set",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        self._set_axis_limits(fig, 1, 1, pareto_set_orig[:, 0], pareto_set_orig[:, 1])
        fig.update_xaxes(title_text="$x_1$", row=1, col=1)
        fig.update_yaxes(title_text="$x_2$", row=1, col=1)
        self._add_description(
            fig,
            1,
            1,
            "Original decision variables showing trade-offs between solutions",
        )

        # Objective space (Row 1, Col 2)
        fig.add_trace(
            go.Scatter(
                x=pareto_front_orig[:, 0],
                y=pareto_front_orig[:, 1],
                mode="markers",
                marker=dict(
                    size=7,
                    opacity=0.8,
                    color="#2ecc71",  # Green
                    line=dict(width=1, color="#2c3e50"),
                ),
                name="Pareto Front",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        self._set_axis_limits(
            fig, 1, 2, pareto_front_orig[:, 0], pareto_front_orig[:, 1]
        )
        fig.update_xaxes(title_text="$f_1$", row=1, col=2)
        fig.update_yaxes(title_text="$f_2$", row=1, col=2)
        self._add_description(
            fig, 1, 2, "Objective space visualization of Pareto optimal solutions"
        )

    def _add_normalized_spaces(self, fig: go.Figure):
        """Visualize normalized decision and objective spaces."""
        # Retrieve normalized data
        norm_x1_all = self._f1_rel_data["norm_x1_all"]
        norm_x2_all = self._f1_rel_data["norm_x2_all"]
        norm_f1_all = self._f1_rel_data["norm_f1_all"]
        norm_f2_all = self._f1_rel_data["norm_f2_all"]

        # Normalized Decision Space (Row 2, Col 1)
        fig.add_trace(
            go.Scatter(
                x=norm_x1_all,
                y=norm_x2_all,
                mode="markers",
                marker=dict(
                    size=6,
                    opacity=0.7,
                    color="#3498db",  # Blue
                    symbol="diamond",
                ),
                name="Norm Pareto Set",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        self._set_axis_limits(fig, 2, 1, norm_x1_all, norm_x2_all)
        fig.update_xaxes(title_text="Norm $x_1$", row=2, col=1)
        fig.update_yaxes(title_text="Norm $x_2$", row=2, col=1)
        self._add_description(
            fig, 2, 1, "Decision variables scaled to [0,1] for comparative analysis"
        )

        # Normalized Objective Space (Row 2, Col 2)
        fig.add_trace(
            go.Scatter(
                x=norm_f1_all,
                y=norm_f2_all,
                mode="markers",
                marker=dict(
                    size=6,
                    opacity=0.7,
                    color="#2ecc71",  # Green
                    symbol="diamond",
                ),
                name="Norm Pareto Front",
                showlegend=True,
            ),
            row=2,
            col=2,
        )
        self._set_axis_limits(fig, 2, 2, norm_f1_all, norm_f2_all)
        fig.update_xaxes(title_text="Norm $f_1$", row=2, col=2)
        fig.update_yaxes(title_text="Norm $f_2$", row=2, col=2)
        self._add_description(
            fig,
            2,
            2,
            "Objective functions normalized for fair comparison across solutions",
        )

    def _add_parallel_coordinates(self, fig: go.Figure):
        """Add parallel coordinates plot for combined variables."""
        norm_x1_all = self._f1_rel_data["norm_x1_all"]
        norm_x2_all = self._f1_rel_data["norm_x2_all"]
        norm_f1_all = self._f1_rel_data["norm_f1_all"]
        norm_f2_all = self._f1_rel_data["norm_f2_all"]

        data = np.hstack(
            (
                norm_x1_all.reshape(-1, 1),
                norm_x2_all.reshape(-1, 1),
                norm_f1_all.reshape(-1, 1),
                norm_f2_all.reshape(-1, 1),
            )
        )

        dims = [
            dict(label="x₁", values=data[:, 0]),
            dict(label="x₂", values=data[:, 1]),
            dict(label="f₁", values=data[:, 2]),
            dict(label="f₂", values=data[:, 3]),
        ]

        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=data[:, 2],
                    colorscale="Viridis",
                    showscale=True,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(
                        title="f₁", thickness=15, len=0.5, yanchor="middle", y=0.5
                    ),
                ),
                dimensions=dims,
            ),
            row=1,
            col=3,
        )
        self._add_description(
            fig,
            1,
            3,
            "Multivariate analysis showing relationships across all dimensions",
        )

    def _add_x1_x2_interpolation(self, fig: go.Figure):
        """
        Add a plot showing the interpolation between x1 and x2.
        """
        data = self._x1_x2_interp_data
        norm_x1_all = data["norm_x1_all"]
        norm_x2_all = data["norm_x2_all"]
        norm_x1_unique = data["norm_x1_unique_for_spline"]
        norm_x2_for_unique_x1 = data["norm_x2_for_spline"]

        can_interpolate_pchip_linear = len(norm_x1_unique) > 1
        can_interpolate_quadratic = len(norm_x1_unique) > 2
        can_interpolate_cubic_spline = len(norm_x1_unique) > 3

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=norm_x1_all,
                y=norm_x2_all,
                mode="markers",
                marker=dict(
                    color="#9b59b6",  # Purple
                    size=7,
                    opacity=0.7,
                    line=dict(width=1, color="#2c3e50"),
                ),
                name="Data Points",
                showlegend=False,
            ),
            row=2,
            col=3,
        )

        x1_interp_norm = np.linspace(0, 1, 100)

        for interp_name in ["Pchip", "Cubic Spline", "Linear", "Quadratic"]:
            interp_func = None
            interpolation_possible = False

            if interp_name == "Pchip" and can_interpolate_pchip_linear:
                interp_func = PchipInterpolator(norm_x1_unique, norm_x2_for_unique_x1)
                interpolation_possible = True
            elif interp_name == "Cubic Spline" and can_interpolate_cubic_spline:
                interp_func = CubicSpline(norm_x1_unique, norm_x2_for_unique_x1)
                interpolation_possible = True
            elif interp_name == "Linear" and can_interpolate_pchip_linear:
                interp_func = interp1d(
                    norm_x1_unique,
                    norm_x2_for_unique_x1,
                    kind="linear",
                    fill_value="extrapolate",
                )
                interpolation_possible = True
            elif interp_name == "Quadratic" and can_interpolate_quadratic:
                interp_func = interp1d(
                    norm_x1_unique,
                    norm_x2_for_unique_x1,
                    kind="quadratic",
                    fill_value="extrapolate",
                )
                interpolation_possible = True

            if interpolation_possible and interp_func:
                try:
                    x2_interp_norm = interp_func(x1_interp_norm)
                    # Control legend appearance
                    show_in_legend = interp_name not in self._seen_interp_methods
                    if show_in_legend:
                        self._seen_interp_methods.add(interp_name)

                    fig.add_trace(
                        go.Scatter(
                            x=x1_interp_norm,
                            y=x2_interp_norm,
                            mode="lines",
                            line=dict(
                                color=self._interp_colors[interp_name],
                                width=3,
                                dash="solid" if interp_name == "Pchip" else "dash",
                            ),
                            name=interp_name,
                            legendgroup=interp_name,
                            showlegend=show_in_legend,
                        ),
                        row=2,
                        col=3,
                    )
                except ValueError:
                    continue

        self._set_axis_limits(
            fig, 2, 3, np.array([0, 1]), np.array([0, 1]), padding=0.01
        )
        fig.update_xaxes(title_text="Normalized $x_1$", row=2, col=3)
        fig.update_yaxes(title_text="Normalized $x_2$", row=2, col=3)

        description_suffix = ""
        if not can_interpolate_pchip_linear:
            description_suffix = " (Not enough data for interpolation)"
        elif not can_interpolate_quadratic:
            description_suffix = " (Not enough data for quadratic/cubic splines)"
        elif not can_interpolate_cubic_spline:
            description_suffix = " (Not enough data for cubic splines)"

        self._add_description(
            fig,
            2,
            3,
            "Relationship between decision variables with interpolation models"
            + description_suffix,
        )

    def _add_f1_relationships(self, fig: go.Figure):
        """
        Add f₁ relationships with f₂, x₁, x₂ using different interpolation methods.
        Now with three separate plots in row 3
        """
        data = self._f1_rel_data

        norm_f1_all = data["norm_f1_all"]
        norm_f2_all = data["norm_f2_all"]
        norm_x1_all = data["norm_x1_all"]
        norm_x2_all = data["norm_x2_all"]

        norm_f1_unique = data["norm_f1_unique_for_spline"]
        norm_f2_for_spline = data["norm_f2_for_spline"]
        norm_x1_for_spline = data["norm_x1_for_spline"]
        norm_x2_for_spline = data["norm_x2_for_spline"]

        plot_configs = [
            (norm_f2_all, "$f_2$", "#e74c3c", 1),  # Red
            (norm_x1_all, "$x_1$", "#f39c12", 2),  # Orange
            (norm_x2_all, "$x_2$", "#16a085", 3),  # Teal
        ]

        can_interpolate_pchip_linear = len(norm_f1_unique) > 1
        can_interpolate_quadratic = len(norm_f1_unique) > 2
        can_interpolate_cubic_spline = len(norm_f1_unique) > 3

        # Create three separate subplots for the relationships
        for i, (norm_y_all_points, label, scatter_color, col) in enumerate(
            plot_configs
        ):
            # Add scatter points to each subplot
            fig.add_trace(
                go.Scatter(
                    x=norm_f1_all,
                    y=norm_y_all_points,
                    mode="markers",
                    marker=dict(
                        color=scatter_color,
                        size=6,
                        opacity=0.7,
                        line=dict(width=1, color="#2c3e50"),
                    ),
                    name=f"{label} Data",
                    showlegend=False,
                ),
                row=3,
                col=i + 1,
            )

            current_norm_y_for_spline = {
                "$f_2$": norm_f2_for_spline,
                "$x_1$": norm_x1_for_spline,
                "$x_2$": norm_x2_for_spline,
            }.get(label)

            f1_interp_norm = np.linspace(0, 1, 100)

            for interp_name in ["Pchip", "Cubic Spline", "Linear", "Quadratic"]:
                interp_func = None
                interpolation_possible = False

                if interp_name == "Pchip" and can_interpolate_pchip_linear:
                    interp_func = PchipInterpolator(
                        norm_f1_unique, current_norm_y_for_spline
                    )
                    interpolation_possible = True
                elif interp_name == "Cubic Spline" and can_interpolate_cubic_spline:
                    interp_func = CubicSpline(norm_f1_unique, current_norm_y_for_spline)
                    interpolation_possible = True
                elif interp_name == "Linear" and can_interpolate_pchip_linear:
                    interp_func = interp1d(
                        norm_f1_unique,
                        current_norm_y_for_spline,
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    interpolation_possible = True
                elif interp_name == "Quadratic" and can_interpolate_quadratic:
                    interp_func = interp1d(
                        norm_f1_unique,
                        current_norm_y_for_spline,
                        kind="quadratic",
                        fill_value="extrapolate",
                    )
                    interpolation_possible = True

                if interpolation_possible and interp_func:
                    try:
                        y_interp_norm = interp_func(f1_interp_norm)
                        # Control legend appearance - only show once per method
                        show_in_legend = interp_name not in self._seen_interp_methods
                        if show_in_legend:
                            self._seen_interp_methods.add(interp_name)

                        fig.add_trace(
                            go.Scatter(
                                x=f1_interp_norm,
                                y=y_interp_norm,
                                mode="lines",
                                line=dict(
                                    color=self._interp_colors[interp_name],
                                    width=2.5,
                                    dash="solid" if interp_name == "Pchip" else "dash",
                                ),
                                name=interp_name,
                                legendgroup=interp_name,
                                showlegend=show_in_legend,
                            ),
                            row=3,
                            col=i + 1,
                        )
                    except ValueError:
                        continue

            self._set_axis_limits(
                fig, 3, i + 1, np.array([0, 1]), np.array([0, 1]), padding=0.01
            )
            fig.update_xaxes(title_text="Normalized $f_1$", row=3, col=i + 1)
            fig.update_yaxes(title_text=f"Normalized {label}", row=3, col=i + 1)

            description_suffix = ""
            if not can_interpolate_pchip_linear:
                description_suffix = " (Not enough data for interpolation)"
            elif not can_interpolate_quadratic:
                description_suffix = " (Not enough data for quadratic/cubic splines)"
            elif not can_interpolate_cubic_spline:
                description_suffix = " (Not enough data for cubic splines)"

            self._add_description(
                fig,
                3,
                i + 1,
                f"Relationship between f₁ and {label} with interpolation models"
                + description_suffix,
            )
