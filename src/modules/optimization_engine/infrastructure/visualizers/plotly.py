from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...domain.analyzing.interfaces.base_visualizer import BaseParetoVisualizer


class PlotlyParetoVisualizer(BaseParetoVisualizer):
    """Dashboard for visualizing Pareto set and front with precomputed interpolations."""

    def __init__(self, save_path: Path | None = None):
        super().__init__(save_path)
        self._f1_rel_data: dict | None = None
        self._interp_data: dict | None = None
        self._interp_colors = {
            "Pchip": "#E64A19",  # Deep orange
            "Cubic Spline": "#1976D2",  # Blue
            "Linear": "#43A047",  # Green
            "Quadratic": "#7B1FA2",  # Purple
        }

    def plot(self, f1_rel_data: dict, interp_data: dict) -> None:
        self._f1_rel_data = f1_rel_data
        self._interp_data = interp_data

        # Extract original data for initial plots
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

        # FIXED: Corrected subplot layout with proper types
        fig = make_subplots(
            rows=5,  # Increased to 5 rows
            cols=3,
            specs=[
                # Row 1: Core visualizations
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "parcoords"}],
                # Row 2: Normalized spaces and x1-x2 interpolation
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                # Row 3: f1 relationships
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                # Row 4: f2 relationships
                [{"type": "scatter"}, {"type": "scatter"}, None],
                # Row 5: 3D visualizations
                [{"type": "scatter3d"}, {"type": "scatter3d"}, None],
            ],
            subplot_titles=[
                # Row 1
                "Decision Space ($x_1$ vs $x_2$)",
                "Objective Space ($f_1$ vs $f_2$)",
                "Parallel Coordinates",
                # Row 2
                "Normalized Decision Space",
                "Normalized Objective Space",
                "$x_1$ vs $x_2$ (Interpolations)",
                # Row 3
                "$f_1$ vs $f_2$ (Interpolations)",
                "$f_1$ vs $x_1$ (Interpolations)",
                "$f_1$ vs $x_2$ (Interpolations)",
                # Row 4
                "$f_2$ vs $x_1$ (Interpolations)",
                "$f_2$ vs $x_2$ (Interpolations)",
                None,
                # Row 5
                "3D: $f_1$, $f_2$, $x_1$",
                "3D: $f_1$, $f_2$, $x_2$",
                None,
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.1,
            column_widths=[0.3, 0.3, 0.4],
            row_heights=[0.15, 0.15, 0.15, 0.15, 0.4],  # Adjusted row heights
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text="Enhanced Pareto Optimization Dashboard", x=0.5, font=dict(size=24)
            ),
            height=2200,  # Increased height for additional row
            width=1800,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.12,  # Adjusted for extra row
                xanchor="center",
                x=0.5,
                title=dict(text="Interpolation Methods", font=dict(size=14)),
            ),
            margin=dict(t=100, b=100, l=50, r=50),
            font=dict(family="Arial", size=12),
        )

        # Add all plots
        self._add_decision_objective_spaces(fig, pareto_set_orig, pareto_front_orig)
        self._add_normalized_spaces(fig)
        self._add_parallel_coordinates(fig)
        self._add_x1_x2_interpolation(fig)
        self._add_f1_relationships(fig)
        self._add_f2_relationships(fig)
        self._add_3d_visualizations(fig)  # Now in row 5

        fig.show()

    def _set_axis_limits(self, fig, row, col, x_data, y_data, padding=0.05):
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
        row_positions = {1: 0.92, 2: 0.72, 3: 0.52, 4: 0.32}
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
        # Decision space (Row 1, Col 1)
        fig.add_trace(
            go.Scatter(
                x=pareto_set_orig[:, 0],
                y=pareto_set_orig[:, 1],
                mode="markers",
                marker=dict(size=7, opacity=0.8, color="#3498db"),
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
            fig, 1, 1, "Original decision variables showing trade-offs"
        )

        # Objective space (Row 1, Col 2)
        fig.add_trace(
            go.Scatter(
                x=pareto_front_orig[:, 0],
                y=pareto_front_orig[:, 1],
                mode="markers",
                marker=dict(size=7, opacity=0.8, color="#2ecc71"),
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
        # Retrieve normalized data
        norm_x1 = self._f1_rel_data["norm_x1_all"]
        norm_x2 = self._f1_rel_data["norm_x2_all"]
        norm_f1 = self._f1_rel_data["norm_f1_all"]
        norm_f2 = self._f1_rel_data["norm_f2_all"]

        # Normalized Decision Space (Row 2, Col 1)
        fig.add_trace(
            go.Scatter(
                x=norm_x1,
                y=norm_x2,
                mode="markers",
                marker=dict(size=6, opacity=0.7, color="#3498db", symbol="diamond"),
                name="Norm Pareto Set",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        self._set_axis_limits(fig, 2, 1, norm_x1, norm_x2)
        fig.update_xaxes(title_text="Norm $x_1$", row=2, col=1)
        fig.update_yaxes(title_text="Norm $x_2$", row=2, col=1)
        self._add_description(fig, 2, 1, "Decision variables scaled to [0,1]")

        # Normalized Objective Space (Row 2, Col 2)
        fig.add_trace(
            go.Scatter(
                x=norm_f1,
                y=norm_f2,
                mode="markers",
                marker=dict(size=6, opacity=0.7, color="#2ecc71", symbol="diamond"),
                name="Norm Pareto Front",
                showlegend=True,
            ),
            row=2,
            col=2,
        )
        self._set_axis_limits(fig, 2, 2, norm_f1, norm_f2)
        fig.update_xaxes(title_text="Norm $f_1$", row=2, col=2)
        fig.update_yaxes(title_text="Norm $f_2$", row=2, col=2)
        self._add_description(
            fig, 2, 2, "Objective functions normalized for comparison"
        )

    def _add_parallel_coordinates(self, fig: go.Figure):
        norm_x1 = self._f1_rel_data["norm_x1_all"]
        norm_x2 = self._f1_rel_data["norm_x2_all"]
        norm_f1 = self._f1_rel_data["norm_f1_all"]
        norm_f2 = self._f1_rel_data["norm_f2_all"]

        data = np.hstack(
            (
                norm_x1.reshape(-1, 1),
                norm_x2.reshape(-1, 1),
                norm_f1.reshape(-1, 1),
                norm_f2.reshape(-1, 1),
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
        self._add_description(fig, 1, 3, "Multivariate analysis across all dimensions")

    def _add_x1_x2_interpolation(self, fig: go.Figure):
        data = self._interp_data
        norm_x1 = data["norm_x1_all"]
        norm_x2 = data["norm_x2_all"]
        interpolations = data["interpolations"]["x1_vs_x2"]

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=norm_x1,
                y=norm_x2,
                mode="markers",
                marker=dict(color="#9b59b6", size=7, opacity=0.7),
                name="Data Points",
                showlegend=False,
            ),
            row=2,
            col=3,
        )

        # FIXED: Legend handling - add all methods with proper grouping
        for method_name, (x_grid, y_grid) in interpolations.items():
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    line=dict(
                        color=self._interp_colors[method_name],
                        width=3,
                        dash="solid" if method_name == "Pchip" else "dash",
                    ),
                    name=method_name,
                    legendgroup=method_name,  # Group by method
                    showlegend=True,  # Always show in legend
                ),
                row=2,
                col=3,
            )

        self._set_axis_limits(
            fig, 2, 3, np.array([0, 1]), np.array([0, 1]), padding=0.01
        )
        fig.update_xaxes(title_text="Normalized $x_1$", row=2, col=3)
        fig.update_yaxes(title_text="Normalized $x_2$", row=2, col=3)

        num_points = len(data.get("norm_x1_unique", []))
        suffix = self._get_interp_suffix(num_points)
        self._add_description(
            fig, 2, 3, "Relationship between decision variables" + suffix
        )

    def _add_f1_relationships(self, fig: go.Figure):
        data = self._f1_rel_data
        norm_f1 = data["norm_f1_all"]
        interpolations = data["interpolations"]

        # Define plot configurations
        plot_configs = [
            (
                norm_f1,
                data["norm_f2_all"],
                "$f_2$",
                "#e74c3c",
                1,
                interpolations["f1_vs_f2"],
                3,
            ),
            (
                norm_f1,
                data["norm_x1_all"],
                "$x_1$",
                "#f39c12",
                2,
                interpolations["f1_vs_x1"],
                3,
            ),
            (
                norm_f1,
                data["norm_x2_all"],
                "$x_2$",
                "#16a085",
                3,
                interpolations["f1_vs_x2"],
                3,
            ),
        ]

        for i, (x_data, y_data, label, color, col, interp_data, row) in enumerate(
            plot_configs
        ):
            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="markers",
                    marker=dict(color=color, size=6, opacity=0.7),
                    name=f"{label} Data",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # FIXED: Add interpolations with proper legend grouping
            for method_name, (x_grid, y_grid) in interp_data.items():
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=y_grid,
                        mode="lines",
                        line=dict(
                            color=self._interp_colors[method_name],
                            width=2.5,
                            dash="solid" if method_name == "Pchip" else "dash",
                        ),
                        name=method_name,
                        legendgroup=method_name,  # Group by method
                        showlegend=False,  # Don't show again in legend
                    ),
                    row=row,
                    col=col,
                )

            self._set_axis_limits(
                fig, row, col, np.array([0, 1]), np.array([0, 1]), padding=0.01
            )
            fig.update_xaxes(title_text="Normalized $f_1$", row=row, col=col)
            fig.update_yaxes(title_text=f"Normalized {label}", row=row, col=col)

            num_points = len(data.get("norm_f1_unique_for_spline", []))
            suffix = self._get_interp_suffix(num_points)
            self._add_description(
                fig, row, col, f"Relationship between f₁ and {label}" + suffix
            )

    def _add_f2_relationships(self, fig: go.Figure):
        data = self._interp_data
        norm_f2 = data["norm_f2_all"]
        interpolations = data["interpolations"]

        # Define plot configurations - now in row 4
        plot_configs = [
            (
                norm_f2,
                data["norm_x1_all"],
                "$x_1$",
                "#f39c12",
                1,
                interpolations["f2_vs_x1"],
                4,
            ),
            (
                norm_f2,
                data["norm_x2_all"],
                "$x_2$",
                "#16a085",
                2,
                interpolations["f2_vs_x2"],
                4,
            ),
        ]

        for i, (x_data, y_data, label, color, col, interp_data, row) in enumerate(
            plot_configs
        ):
            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="markers",
                    marker=dict(color=color, size=6, opacity=0.7),
                    name=f"{label} Data",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # FIXED: Add interpolations with proper legend grouping
            for method_name, (x_grid, y_grid) in interp_data.items():
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=y_grid,
                        mode="lines",
                        line=dict(
                            color=self._interp_colors[method_name],
                            width=2.5,
                            dash="solid" if method_name == "Pchip" else "dash",
                        ),
                        name=method_name,
                        legendgroup=method_name,  # Group by method
                        showlegend=False,  # Don't show again in legend
                    ),
                    row=row,
                    col=col,
                )

            self._set_axis_limits(
                fig, row, col, np.array([0, 1]), np.array([0, 1]), padding=0.01
            )
            fig.update_xaxes(title_text="Normalized $f_2$", row=row, col=col)
            fig.update_yaxes(title_text=f"Normalized {label}", row=row, col=col)

            num_points = len(data.get("norm_f2_unique_for_spline", []))
            suffix = self._get_interp_suffix(num_points)
            self._add_description(
                fig, row, col, f"Relationship between f₂ and {label}" + suffix
            )

    def _add_3d_visualizations(self, fig: go.Figure):
        # Retrieve normalized data
        norm_f1 = self._f1_rel_data["norm_f1_all"]
        norm_f2 = self._f1_rel_data["norm_f2_all"]
        norm_x1 = self._f1_rel_data["norm_x1_all"]
        norm_x2 = self._f1_rel_data["norm_x2_all"]
        colorscale = "Viridis"

        # FIXED: Moved to row 5
        # 3D: f1, f2, x1 (Row 5, Col 1)
        fig.add_trace(
            go.Scatter3d(
                x=norm_f1,
                y=norm_f2,
                z=norm_x1,
                mode="markers",
                marker=dict(
                    size=5,
                    color=norm_x1,
                    colorscale=colorscale,
                    opacity=0.8,
                    colorbar=dict(title="x₁", thickness=20),
                ),
                name="f1-f2-x1",
            ),
            row=5,
            col=1,  # Changed to row 5
        )

        # 3D: f1, f2, x2 (Row 5, Col 2)
        fig.add_trace(
            go.Scatter3d(
                x=norm_f1,
                y=norm_f2,
                z=norm_x2,
                mode="markers",
                marker=dict(
                    size=5,
                    color=norm_x2,
                    colorscale=colorscale,
                    opacity=0.8,
                    colorbar=dict(title="x₂", thickness=20),
                ),
                name="f1-f2-x2",
            ),
            row=5,
            col=2,  # Changed to row 5
        )

        # Update scenes
        fig.update_scenes(
            row=5,
            col=1,  # Updated row
            xaxis_title_text="Normalized $f_1$",
            yaxis_title_text="Normalized $f_2$",
            zaxis_title_text="Normalized $x_1$",
        )
        fig.update_scenes(
            row=5,
            col=2,  # Updated row
            xaxis_title_text="Normalized $f_1$",
            yaxis_title_text="Normalized $f_2$",
            zaxis_title_text="Normalized $x_2$",
        )

        # Add descriptions
        self._add_description(fig, 5, 1, "3D visualization: f₁, f₂ and x₁")
        self._add_description(fig, 5, 2, "3D visualization: f₁, f₂ and x₂")

    def _get_interp_suffix(self, num_points: int) -> str:
        """Generate appropriate suffix based on available points"""
        if num_points < 2:
            return " (Not enough data for interpolation)"
        elif num_points < 3:
            return " (Not enough data for quadratic/cubic splines)"
        elif num_points < 4:
            return " (Not enough data for cubic splines)"
        return ""
