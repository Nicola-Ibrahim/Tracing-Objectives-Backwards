from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d

from ...domain.analyzing.interfaces.base_visualizer import BaseParetoVisualizer


class PlotlyParetoVisualizer(BaseParetoVisualizer):
    """Dashboard for visualizing Pareto set and front using Plotly."""

    def __init__(self, save_path: Path | None = None):
        super().__init__(save_path)
        # These will now be populated by the 'plot' method with pre-prepared data
        self._f1_rel_data: dict | None = None
        self._x1_x2_interp_data: dict | None = None

    def plot(self, f1_rel_data: dict, x1_x2_interp_data: dict) -> None:
        """
        Generate an interactive dashboard with multiple Pareto visualizations
        using pre-prepared data dictionaries.
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

        # 2. Setup subplot layout
        fig = make_subplots(
            rows=4,
            cols=3,
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "parcoords", "colspan": 2}, None, {"type": "scatter"}],
            ],
            subplot_titles=[
                "Decision Space ($x_1$ vs $x_2$)",
                "Objective Space ($f_1$ vs $f_2$)",
                "Decision vs Objective",
                "Normalized Decision Space",
                "Normalized Objective Space",
                "Normalized Decision vs Objective",
                "$f_1$ vs $f_2$ (Normalized Interpolations)",
                "$f_1$ vs $x_1$ (Normalized Interpolations)",
                "$f_1$ vs $x_2$ (Normalized Interpolations)",
                "Parallel Coordinates",
                "Interpolation between $x_1$ and $x_2$ (Normalized Interpolations)",
            ],
            horizontal_spacing=0.05,
            vertical_spacing=0.07,
        )

        # 3. Update overall layout
        fig.update_layout(
            title_text="Pareto Optimization Analysis Dashboard",
            height=1800,
            width=1600,
            showlegend=True,
            template="plotly_white",
        )

        # 4. Add plots using prepared data
        self._add_decision_objective_spaces(fig, pareto_set_orig, pareto_front_orig)
        # These methods now directly use the 'norm_all' data from _f1_rel_data and _x1_x2_interp_data
        self._add_normalized_spaces(fig)
        self._add_f1_relationships(fig)
        self._add_parallel_coordinates(fig)
        self._add_x1_x2_interpolation(fig)

        # 5. Save and Show
        self.save_path.mkdir(parents=True, exist_ok=True)
        fig.write_image(
            file=self.save_path / "pareto_dashboard.png",
            width=1600,
            height=1800,
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
        """Add description text annotation below a subplot."""
        axis_num = col + (row - 1) * 3
        xref = "x domain" if axis_num == 1 else f"x{axis_num} domain"
        yref = "y domain" if axis_num == 1 else f"y{axis_num} domain"

        fig.add_annotation(
            text=text,
            x=0.5,
            y=-0.20,
            xref=xref,
            yref=yref,
            showarrow=False,
            font=dict(size=11, color="grey"),
            align="center",
        )

    def _add_decision_objective_spaces(
        self, fig: go.Figure, pareto_set_orig: np.ndarray, pareto_front_orig: np.ndarray
    ):
        """Visualize Pareto set and front in original space."""
        markers = dict(size=6, opacity=0.7)

        # Decision space
        fig.add_trace(
            go.Scatter(
                x=pareto_set_orig[:, 0],
                y=pareto_set_orig[:, 1],
                mode="markers",
                marker={**markers, "color": "blue"},
                name="Pareto Set",
            ),
            row=1,
            col=1,
        )
        self._set_axis_limits(fig, 1, 1, pareto_set_orig[:, 0], pareto_set_orig[:, 1])
        fig.update_xaxes(title_text="$x_1$", row=1, col=1)
        fig.update_yaxes(title_text="$x_2$", row=1, col=1)
        self._add_description(
            fig, 1, 1, "Shows the Pareto set in the original decision variable space."
        )

        # Objective space
        fig.add_trace(
            go.Scatter(
                x=pareto_front_orig[:, 0],
                y=pareto_front_orig[:, 1],
                mode="markers",
                marker={**markers, "color": "green"},
                name="Pareto Front",
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
            fig,
            1,
            2,
            "Shows the Pareto front in the original objective function space.",
        )

        # Decision vs Objective
        fig.add_trace(
            go.Scatter(
                x=pareto_set_orig[:, 0],
                y=pareto_front_orig[:, 0],
                mode="markers",
                marker={**markers, "color": "purple"},
                name="x₁ vs f₁",
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=pareto_set_orig[:, 1],
                y=pareto_front_orig[:, 1],
                mode="markers",
                marker={**markers, "color": "orange"},
                name="x₂ vs f₂",
            ),
            row=1,
            col=3,
        )
        combined_x = np.concatenate([pareto_set_orig[:, 0], pareto_set_orig[:, 1]])
        combined_y = np.concatenate([pareto_front_orig[:, 0], pareto_front_orig[:, 1]])
        self._set_axis_limits(fig, 1, 3, combined_x, combined_y)
        fig.update_xaxes(title_text="Decision variables ($x$)", row=1, col=3)
        fig.update_yaxes(title_text="Objective values ($f$)", row=1, col=3)
        self._add_description(
            fig,
            1,
            3,
            "Relationship between decision variables and their corresponding objectives.",
        )

    def _add_normalized_spaces(self, fig: go.Figure):
        """Visualize normalized decision and objective spaces using pre-prepared normalized data."""
        # Retrieve normalized data from the prepared data dictionaries
        norm_x1_all = self._f1_rel_data["norm_x1_all"]
        norm_x2_all = self._f1_rel_data["norm_x2_all"]
        norm_f1_all = self._f1_rel_data["norm_f1_all"]
        norm_f2_all = self._f1_rel_data["norm_f2_all"]

        markers = dict(size=6, opacity=0.7)

        # Normalized Decision Space
        fig.add_trace(
            go.Scatter(
                x=norm_x1_all,
                y=norm_x2_all,
                mode="markers",
                marker={**markers, "color": "blue"},
                name="Norm Pareto Set",
            ),
            row=2,
            col=1,
        )
        self._set_axis_limits(fig, 2, 1, norm_x1_all, norm_x2_all)
        fig.update_xaxes(title_text="Norm $x_1$", row=2, col=1)
        fig.update_yaxes(title_text="Norm $x_2$", row=2, col=1)
        self._add_description(
            fig,
            2,
            1,
            "Normalized decision space showing scaled decision variables between 0 and 1.",
        )

        # Normalized Objective Space
        fig.add_trace(
            go.Scatter(
                x=norm_f1_all,
                y=norm_f2_all,
                mode="markers",
                marker={**markers, "color": "green"},
                name="Norm Pareto Front",
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
            "Normalized objective space showing scaled objective values between 0 and 1.",
        )

        # Combined Normalized Decision vs Objective
        fig.add_trace(
            go.Scatter(
                x=norm_x1_all,
                y=norm_f1_all,
                mode="markers",
                marker={**markers, "color": "purple"},
                name="Norm x₁ vs f₁",
            ),
            row=2,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=norm_x2_all,
                y=norm_f2_all,
                mode="markers",
                marker={**markers, "color": "orange"},
                name="Norm x₂ vs f₂",
            ),
            row=2,
            col=3,
        )
        combined_x = np.concatenate([norm_x1_all, norm_x2_all])
        combined_y = np.concatenate([norm_f1_all, norm_f2_all])
        self._set_axis_limits(fig, 2, 3, combined_x, combined_y)
        fig.update_xaxes(title_text="Normalized $x$", row=2, col=3)
        fig.update_yaxes(title_text="Normalized $f$", row=2, col=3)
        self._add_description(
            fig,
            2,
            3,
            "Normalized relationship between decision variables and objectives.",
        )

    def _add_f1_relationships(self, fig: go.Figure):
        """
        Add f₁ relationships with f₂, x₁, x₂ using different interpolation methods.
        Both scatter points and interpolation lines are in normalized space.
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

        interpolation_methods = [
            ("Pchip", "red", "solid"),
            ("Cubic Spline", "blue", "dash"),
            ("Linear", "green", "dot"),
            ("Quadratic", "purple", "dashdot"),
        ]

        # Check if enough unique points exist for higher-order interpolations
        can_interpolate_pchip_linear = (
            len(norm_f1_unique) > 1
        )  # Needs at least 2 points
        can_interpolate_quadratic = len(norm_f1_unique) > 2  # Needs at least 3 points
        can_interpolate_cubic_spline = (
            len(norm_f1_unique) > 3
        )  # Needs at least 4 points

        for i, (norm_y_all_points, label, scatter_color) in enumerate(
            [
                (norm_f2_all, "$f_2$", "purple"),
                (norm_x1_all, "$x_1$", "orange"),
                (norm_x2_all, "$x_2$", "brown"),
            ]
        ):
            # Plot normalized data points for scatter
            fig.add_trace(
                go.Scatter(
                    x=norm_f1_all,
                    y=norm_y_all_points,
                    mode="markers",
                    marker=dict(color=scatter_color, size=6, opacity=0.7),
                    name=f"Normalized Data Points {label}",
                    showlegend=False,
                ),
                row=3,
                col=i + 1,
            )

            # Dynamically get the correct y-values for the spline based on current label
            current_norm_y_for_spline = {
                "$f_2$": norm_f2_for_spline,
                "$x_1$": norm_x1_for_spline,
                "$x_2$": norm_x2_for_spline,
            }.get(label)

            # Generate points for the interpolated curve in normalized space (0 to 1)
            f1_interp_norm = np.linspace(0, 1, 100)

            # Add each interpolation method
            for interp_name, line_color, line_dash in interpolation_methods:
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
                        fig.add_trace(
                            go.Scatter(
                                x=f1_interp_norm,
                                y=y_interp_norm,
                                mode="lines",
                                line=dict(color=line_color, dash=line_dash, width=2),
                                name=f"{interp_name} Interpolation {label}",
                                showlegend=(
                                    i == 0
                                ),  # Only show legend for the first column to avoid redundancy
                            ),
                            row=3,
                            col=i + 1,
                        )
                    except ValueError as e:
                        # Catch potential errors if extrapolation fails or data is ill-conditioned
                        print(f"Skipping {interp_name} for {label} due to error: {e}")
                        pass

            # Set axis limits to [0, 1] for normalized plots
            self._set_axis_limits(
                fig, 3, i + 1, np.array([0, 1]), np.array([0, 1]), padding=0.01
            )
            fig.update_xaxes(title_text="Normalized $f_1$", row=3, col=i + 1)
            fig.update_yaxes(title_text=f"Normalized {label}", row=3, col=i + 1)

            description_suffix = ""
            if not can_interpolate_pchip_linear:
                description_suffix = " (Not enough data for interpolation.)"
            elif not can_interpolate_quadratic:
                description_suffix = " (Not enough data for quadratic/cubic splines.)"
            elif not can_interpolate_cubic_spline:
                description_suffix = " (Not enough data for cubic splines.)"

            self._add_description(
                fig,
                3,
                i + 1,
                f"Shows normalized {label} vs $f_1$ with various interpolation methods."
                + description_suffix,
            )

    def _add_parallel_coordinates(self, fig: go.Figure):
        """Add parallel coordinates plot for combined variables using pre-prepared normalized data."""
        # Retrieve normalized data from the prepared data dictionaries
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
            dict(label="x1", values=data[:, 0]),
            dict(label="x2", values=data[:, 1]),
            dict(label="f1", values=data[:, 2]),
            dict(label="f2", values=data[:, 3]),
        ]

        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=data[:, 2],
                    colorscale="Viridis",
                    showscale=False,  # Color by normalized f1
                ),
                dimensions=dims,
            ),
            row=4,
            col=1,
        )
        self._add_description(
            fig,
            4,
            1,
            "Parallel coordinates showing all decision variables and objectives to visualize trade-offs.",
        )

    def _add_x1_x2_interpolation(self, fig: go.Figure):
        """
        Add a plot showing the interpolation between x1 and x2 using different interpolation methods.
        Both scatter points and interpolation lines are in normalized space.
        """
        data = self._x1_x2_interp_data

        norm_x1_all = data["norm_x1_all"]
        norm_x2_all = data["norm_x2_all"]

        norm_x1_unique = data["norm_x1_unique_for_spline"]
        norm_x2_for_unique_x1 = data["norm_x2_for_spline"]

        interpolation_methods = [
            ("Pchip", "red", "solid"),
            ("Cubic Spline", "blue", "dash"),
            ("Linear", "green", "dot"),
            ("Quadratic", "purple", "dashdot"),
        ]

        can_interpolate_pchip_linear = len(norm_x1_unique) > 1
        can_interpolate_quadratic = len(norm_x1_unique) > 2
        can_interpolate_cubic_spline = len(norm_x1_unique) > 3

        # Plot normalized data points for scatter
        fig.add_trace(
            go.Scatter(
                x=norm_x1_all,
                y=norm_x2_all,
                mode="markers",
                marker=dict(color="darkcyan", size=6, opacity=0.7),
                name="Normalized Data Points ($x_1$ vs $x_2$)",
                showlegend=False,
            ),
            row=4,
            col=3,
        )

        # Generate points for the interpolated curve in normalized space (0 to 1)
        x1_interp_norm = np.linspace(0, 1, 100)

        # Add each interpolation method
        for interp_name, line_color, line_dash in interpolation_methods:
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
                    fig.add_trace(
                        go.Scatter(
                            x=x1_interp_norm,
                            y=x2_interp_norm,
                            mode="lines",
                            line=dict(color=line_color, dash=line_dash, width=2),
                            name=f"{interp_name} Interpolation ($x_1$ vs $x_2$)",
                            showlegend=True,
                        ),
                        row=4,
                        col=3,
                    )
                except ValueError as e:
                    print(f"Skipping {interp_name} for x1 vs x2 due to error: {e}")
                    pass

        self._set_axis_limits(
            fig, 4, 3, np.array([0, 1]), np.array([0, 1]), padding=0.01
        )

        fig.update_xaxes(title_text="Normalized $x_1$", row=4, col=3)
        fig.update_yaxes(title_text="Normalized $x_2$", row=4, col=3)

        description_suffix = ""
        if not can_interpolate_pchip_linear:
            description_suffix = " (Not enough data for interpolation.)"
        elif not can_interpolate_quadratic:
            description_suffix = " (Not enough data for quadratic/cubic splines.)"
        elif not can_interpolate_cubic_spline:
            description_suffix = " (Not enough data for cubic splines.)"

        self._add_description(
            fig,
            4,
            3,
            "Visualizes the relationship and various interpolation methods between normalized decision variables $x_1$ and $x_2$."
            + description_suffix,
        )
