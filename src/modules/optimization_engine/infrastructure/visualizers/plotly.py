from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...domain.analyzing.interfaces.base_visualizer import BaseParetoVisualizer


class PlotlyParetoVisualizer(BaseParetoVisualizer):
    """Dashboard for visualizing Pareto set and front with precomputed interpolations."""

    def __init__(self, save_path: Path | None = None):
        super().__init__(save_path)
        self.f1_data: dict | None = None
        self.interp_data: dict | None = None
        self.interpolation_colors = {
            "Pchip": "#07FF03",  # Deep orange
            "Cubic Spline": "#CD05F9",  # Blue
            "Linear": "#43A047",  # Green
            "Quadratic": "#7B1FA2",  # Purple
            "RBF": "#F30B0B",  # Yellow
            "Nearest Neighbor": "#F57C00",  # Orange
            "Linear ND": "#FFEA07",  # Light blue
        }

    def plot(self, f1_data: dict, interp_data: dict) -> None:
        self.f1_data = f1_data
        self.interp_data = interp_data

        # Prepare original data for visualization
        pareto_set = self._prepare_pareto_set(f1_data)
        pareto_front = self._prepare_pareto_front(f1_data)

        # Create figure with appropriate layout
        fig = self._create_figure_layout()

        # Add all visualization components
        self._add_core_visualizations(fig, pareto_set, pareto_front)
        self._add_normalized_spaces(fig)
        self._add_parallel_coordinates(fig)
        self._add_interpolation_visualizations(fig)
        self._add_3d_visualizations(fig)

        fig.show()

    def _prepare_pareto_set(self, data: dict) -> np.ndarray:
        """Combine decision variables into a 2D array"""
        return np.hstack(
            [
                data["x1_orig"].reshape(-1, 1),
                data["x2_orig"].reshape(-1, 1),
            ]
        )

    def _prepare_pareto_front(self, data: dict) -> np.ndarray:
        """Combine objective functions into a 2D array"""
        return np.hstack(
            [
                data["f1_orig"].reshape(-1, 1),
                data["f2_orig"].reshape(-1, 1),
            ]
        )

    def _create_figure_layout(self) -> go.Figure:
        """Create and configure the figure layout"""
        fig = make_subplots(
            rows=5,
            cols=3,
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "parcoords"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, None],
                [{"type": "scatter3d"}, {"type": "scatter3d"}, None],
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
                "$f_2$ vs $x_1$ (Interpolations)",
                "$f_2$ vs $x_2$ (Interpolations)",
                None,
                "3D: $f_1$, $f_2$, $x_1$",
                "3D: $f_1$, $f_2$, $x_2$",
                None,
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.1,
            column_widths=[0.3, 0.3, 0.4],
            row_heights=[0.15, 0.15, 0.15, 0.15, 0.4],
        )

        fig.update_layout(
            title=dict(
                text="Enhanced Pareto Optimization Dashboard", x=0.5, font=dict(size=24)
            ),
            height=2200,
            width=1800,
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
            font=dict(family="Arial", size=12),
        )
        return fig

    def _add_core_visualizations(
        self, fig: go.Figure, pareto_set: np.ndarray, pareto_front: np.ndarray
    ) -> None:
        """Add decision space and objective space visualizations"""
        # Decision space
        self._add_scatter_plot(
            fig,
            x=pareto_set[:, 0],
            y=pareto_set[:, 1],
            row=1,
            col=1,
            name="Pareto Set",
            color="#3498db",
            title_x="$x_1$",
            title_y="$x_2$",
            description="Original decision variables showing trade-offs",
        )

        # Objective space
        self._add_scatter_plot(
            fig,
            x=pareto_front[:, 0],
            y=pareto_front[:, 1],
            row=1,
            col=2,
            name="Pareto Front",
            color="#3498db",
            title_x="$f_1$",
            title_y="$f_2$",
            description="Objective space visualization of Pareto optimal solutions",
        )

        # Parallel coordinates
        self._add_parallel_coordinates(fig)

    def _add_scatter_plot(
        self,
        fig: go.Figure,
        x: np.ndarray,
        y: np.ndarray,
        row: int,
        col: int,
        name: str,
        color: str,
        title_x: str,
        title_y: str,
        description: str,
        marker_size: int = 7,
        symbol: str = "circle",
        showlegend: bool = True,
    ) -> None:
        """Helper to add standardized scatter plots"""
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=marker_size, opacity=0.8, color=color, symbol=symbol),
                name=name,
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
        self._set_axis_limits(fig, row, col, x, y)
        fig.update_xaxes(title_text=title_x, row=row, col=col)
        fig.update_yaxes(title_text=title_y, row=row, col=col)
        self._add_description(fig, row, col, description)

    def _add_normalized_spaces(self, fig: go.Figure) -> None:
        """Add normalized decision and objective spaces"""
        # Normalized Decision Space
        self._add_scatter_plot(
            fig,
            x=self.f1_data["norm_x1"],
            y=self.f1_data["norm_x2"],
            row=2,
            col=1,
            name="Norm Pareto Set",
            color="#3498db",
            title_x="Norm $x_1$",
            title_y="Norm $x_2$",
            description="Decision variables scaled to [0,1]",
            marker_size=6,
            symbol="diamond",
        )

        # Normalized Objective Space
        self._add_scatter_plot(
            fig,
            x=self.f1_data["norm_f1"],
            y=self.f1_data["norm_f2"],
            row=2,
            col=2,
            name="Norm Pareto Front",
            color="#3498db",
            title_x="Norm $f_1$",
            title_y="Norm $f_2$",
            description="Objective functions normalized for comparison",
            marker_size=6,
            symbol="diamond",
        )

    def _add_parallel_coordinates(self, fig: go.Figure) -> None:
        """Add parallel coordinates plot for multivariate analysis"""
        norm_data = np.hstack(
            (
                self.f1_data["norm_x1"].reshape(-1, 1),
                self.f1_data["norm_x2"].reshape(-1, 1),
                self.f1_data["norm_f1"].reshape(-1, 1),
                self.f1_data["norm_f2"].reshape(-1, 1),
            )
        )

        dimensions = [
            dict(label="x₁", values=norm_data[:, 0]),
            dict(label="x₂", values=norm_data[:, 1]),
            dict(label="f₁", values=norm_data[:, 2]),
            dict(label="f₂", values=norm_data[:, 3]),
        ]

        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=norm_data[:, 2],
                    colorscale="Viridis",
                    showscale=True,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(
                        title="f₁", thickness=15, len=0.5, yanchor="middle", y=0.5
                    ),
                ),
                dimensions=dimensions,
            ),
            row=1,
            col=3,
        )
        self._add_description(fig, 1, 3, "Multivariate analysis across all dimensions")

    def _add_interpolation_visualizations(self, fig: go.Figure) -> None:
        """Add all interpolation-related visualizations"""
        self._add_x1_x2_interpolation(fig)
        self._add_f1_relationships(fig)
        self._add_f2_relationships(fig)

    def _add_x1_x2_interpolation(self, fig: go.Figure) -> None:
        """Visualize interpolation between x1 and x2"""
        norm_x1 = self.interp_data["norm_x1"]
        norm_x2 = self.interp_data["norm_x2"]
        interpolations = self.interp_data["interpolations"]["x1_vs_x2"]

        # Add data points
        self._add_scatter_plot(
            fig,
            x=norm_x1,
            y=norm_x2,
            row=2,
            col=3,
            name="Data Points",
            color="#3498db",
            title_x="Normalized $x_1$",
            title_y="Normalized $x_2$",
            description=self._get_interpolation_description(
                "decision variables", self.interp_data.get("norm_x1_unique", [])
            ),
            showlegend=False,
        )

        # Add interpolation lines
        for method_name, (x_grid, y_grid) in interpolations.items():
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    line=dict(
                        color=self.interpolation_colors[method_name],
                        width=3,
                        dash="solid" if method_name == "Pchip" else "dash",
                    ),
                    name=method_name,
                    legendgroup=method_name,
                    showlegend=True,
                ),
                row=2,
                col=3,
            )

    def _add_f1_relationships(self, fig: go.Figure) -> None:
        """Visualize relationships with f1"""
        norm_f1 = self.f1_data["norm_f1"]
        interpolations = self.f1_data["interpolations"]
        num_points = len(self.f1_data.get("norm_f1_unique", []))

        # f1 vs f2
        self._add_relationship_plot(
            fig,
            x_data=norm_f1,
            y_data=self.f1_data["norm_f2"],
            interpolations=interpolations["f1_vs_f2"],
            row=3,
            col=1,
            y_label="$f_2$",
            color="#3498db",
            description=f"Relationship between f₁ and $f_2${
                self._get_interpolation_suffix(num_points)}",
        )

        # f1 vs x1
        self._add_relationship_plot(
            fig,
            x_data=norm_f1,
            y_data=self.f1_data["norm_x1"],
            interpolations=interpolations["f1_vs_x1"],
            row=3,
            col=2,
            y_label="$x_1$",
            color="#3498db",
            description=f"Relationship between f₁ and $x_1${
                self._get_interpolation_suffix(num_points)}",
        )

        # f1 vs x2
        self._add_relationship_plot(
            fig,
            x_data=norm_f1,
            y_data=self.f1_data["norm_x2"],
            interpolations=interpolations["f1_vs_x2"],
            row=3,
            col=3,
            y_label="$x_2$",
            color="#3498db",
            description=f"Relationship between f₁ and $x_2${
                self._get_interpolation_suffix(num_points)}",
        )

    def _add_f2_relationships(self, fig: go.Figure) -> None:
        """Visualize relationships with f2"""
        norm_f2 = self.interp_data["norm_f2"]
        interpolations = self.interp_data["interpolations"]
        num_points = len(self.interp_data.get("norm_f2_unique", []))

        # f2 vs x1
        self._add_relationship_plot(
            fig,
            x_data=norm_f2,
            y_data=self.interp_data["norm_x1"],
            interpolations=interpolations["f2_vs_x1"],
            row=4,
            col=1,
            y_label="$x_1$",
            color="#3498db",
            description=f"Relationship between f₂ and $x_1${
                self._get_interpolation_suffix(num_points)}",
        )

        # f2 vs x2
        self._add_relationship_plot(
            fig,
            x_data=norm_f2,
            y_data=self.interp_data["norm_x2"],
            interpolations=interpolations["f2_vs_x2"],
            row=4,
            col=2,
            y_label="$x_2$",
            color="#3498db",
            description=f"Relationship between f₂ and $x_2${
                self._get_interpolation_suffix(num_points)}",
        )

    def _add_relationship_plot(
        self,
        fig: go.Figure,
        x_data: np.ndarray,
        y_data: np.ndarray,
        interpolations: dict,
        row: int,
        col: int,
        y_label: str,
        color: str,
        description: str,
    ) -> None:
        """Helper to add standardized relationship plots"""
        # Add data points
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                marker=dict(size=6, opacity=0.7, color=color),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Add interpolation lines
        for method_name, (x_grid, y_grid) in interpolations.items():
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    line=dict(
                        color=self.interpolation_colors[method_name],
                        width=2.5,
                        dash="solid" if method_name == "Pchip" else "dash",
                    ),
                    name=method_name,
                    legendgroup=method_name,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        self._set_axis_limits(
            fig, row, col, np.array([0, 1]), np.array([0, 1]), padding=0.01
        )
        fig.update_xaxes(
            title_text="Normalized $f_1$" if "f1" in y_label else "Normalized $f_2$",
            row=row,
            col=col,
        )
        fig.update_yaxes(title_text=f"Normalized {y_label}", row=row, col=col)
        self._add_description(fig, row, col, description)

    def _add_3d_visualizations(self, fig: go.Figure) -> None:
        """Add 3D visualizations of relationships"""
        # Check if multivariate data exists
        if "multivariate_interpolations" not in self.interp_data:
            return

        # Get multivariate interpolation data
        mv_data = self.interp_data["multivariate_interpolations"]

        # Track which legend items we've added
        added_legend_items = set()

        # f1, f2, x1
        self._add_3d_relationship(
            fig,
            surface_data=mv_data["f1f2_vs_x1"],
            x=self.f1_data["norm_f1"],
            y=self.f1_data["norm_f2"],
            z=self.f1_data["norm_x1"],
            row=5,
            col=1,
            z_title="x_1",
            description="3D: $f_1$, $f_2$ and $x_1$ with interpolation",
            added_legend_items=added_legend_items,  # Pass the tracker
        )

        # f1, f2, x2
        self._add_3d_relationship(
            fig,
            surface_data=mv_data["f1f2_vs_x2"],
            x=self.f1_data["norm_f1"],
            y=self.f1_data["norm_f2"],
            z=self.f1_data["norm_x2"],
            row=5,
            col=2,
            z_title="x_2",
            description="3D: $f_1$, $f_2$ and $x_2$ with interpolation",
            added_legend_items=added_legend_items,  # Pass the same tracker
        )

    def _add_3d_relationship(
        self,
        fig: go.Figure,
        surface_data: dict,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        row: int,
        col: int,
        z_title: str,
        description: str,
        added_legend_items: set,  # Add this parameter to track created legend items
    ) -> None:
        """Add 3D surface plot for multivariate relationships"""
        # Add original points (only show legend for first plot)
        show_legend = "Original Points" not in added_legend_items
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=5, opacity=0.8, color="#1f77b4"),
                name="Original Points",
                legendgroup="points",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        if show_legend:
            added_legend_items.add("Original Points")

        # Add interpolation surfaces
        for method_name, (X_grid, Y_grid, Z_grid) in surface_data.items():
            if method_name in ["Nearest Neighbor", "Linear ND"]:
                # Only create legend item once per method
                show_in_legend = method_name not in added_legend_items

                # Add the actual surface
                fig.add_trace(
                    go.Surface(
                        x=X_grid,
                        y=Y_grid,
                        z=Z_grid,
                        colorscale="Viridis",
                        opacity=0.7,
                        name=method_name,
                        showlegend=show_in_legend,  # Show in legend only once
                        showscale=True,
                        colorbar=dict(title=f"Norm {z_title}"),
                        legendgroup=method_name,
                    ),
                    row=row,
                    col=col,
                )

                if show_in_legend:
                    added_legend_items.add(method_name)

        # Update scene settings
        fig.update_scenes(
            row=row,
            col=col,
            xaxis_title_text="Normalized $f_1$",
            yaxis_title_text="Normalized $f_2$",
            zaxis_title_text=f"Normalized ${z_title}$",
            aspectmode="cube",
        )
        self._add_description(fig, row, col, description)

    def _set_axis_limits(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x_data: np.ndarray,
        y_data: np.ndarray,
        padding: float = 0.05,
    ) -> None:
        """Set axis limits with padding"""
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

    def _add_description(self, fig: go.Figure, row: int, col: int, text: str) -> None:
        """Add description text below subplot"""
        row_positions = {1: 0.92, 2: 0.72, 3: 0.52, 4: 0.32, 5: 0.12}
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

    def _get_interpolation_description(
        self, relationship: str, unique_points: list
    ) -> str:
        """Generate description with interpolation status"""
        return f"Relationship between {relationship}" + self._get_interpolation_suffix(
            len(unique_points)
        )

    def _get_interpolation_suffix(self, num_points: int) -> str:
        """Generate status suffix based on available points"""
        if num_points < 2:
            return " (Not enough data for interpolation)"
        if num_points < 3:
            return " (Not enough data for quadratic/cubic splines)"
        if num_points < 4:
            return " (Not enough data for cubic splines)"
        return ""
