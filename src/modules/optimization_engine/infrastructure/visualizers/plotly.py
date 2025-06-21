from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...domain.analyzing.interfaces.base_visualizer import BaseParetoVisualizer
from .mapper import ParetoVisualizationDTO


class PlotlyParetoVisualizer(BaseParetoVisualizer):
    """
    Dashboard for visualizing Pareto set and front with precomputed interpolations.

    This class orchestrates the creation of a comprehensive Plotly dashboard
    to display multi-objective optimization results, including decision space,
    objective space, parallel coordinates, normalized spaces, and
    various interpolation visualizations.
    """

    # --- Configuration Constants ---
    _INTERPOLATION_COLORS = {
        "Pchip": "#07FF03",
        "Cubic Spline": "#CD05F9",
        "Linear": "#43A047",
        "Quadratic": "#7B1FA2",
        "RBF": "#F30B0B",
        "Nearest Neighbor": "#F57C00",
        "Linear ND": "#FFEA07",
    }

    _SUBPLOT_LAYOUT = {
        "rows": 5,
        "cols": 3,
        "specs": [
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "parcoords"}],  # Row 1
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],  # Row 2
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],  # Row 3
            [
                {"type": "scatter"},
                {"type": "scatter"},
                None,
            ],  # Row 4 (f2_vs_x1, f2_vs_x2) - NO PLOT AT COL 3
            [{"type": "scatter3d"}, {"type": "scatter3d"}, None],  # Row 5
        ],
        "subplot_titles": [
            "Decision Space ($x_1$ vs $x_2$)",
            "Objective Space ($f_1$ vs $f_2$)",
            "Parallel Coordinates",
            "Normalized Decision Space",
            "Normalized Objective Space",
            "$x_1$ vs $x_2$ (Interpolations)",
            "$f_1$ vs $f_2$ (Interpolations)",
            "$f_1$ vs $x_1$ (Interpolations)",
            "$f_1$ vs $x_2$ (Interpolations)",
            "$f_2$ vs $x_1$ (Interpolations)",  # This is at (4,1)
            "$f_2$ vs $x_2$ (Interpolations)",  # This is at (4,2)
            None,  # This is (4,3)
            "3D: $f_1$, $f_2$, $x_1$",
            "3D: $f_1$, $f_2$, $x_2$",
            None,
        ],
        "horizontal_spacing": 0.08,
        "vertical_spacing": 0.1,
        "column_widths": [0.3, 0.3, 0.4],
        "row_heights": [0.15, 0.15, 0.15, 0.15, 0.4],
    }

    _FIGURE_LAYOUT_CONFIG = {
        "title_text": "Enhanced Pareto Optimization Dashboard",
        "title_x": 0.5,
        "title_font_size": 24,
        "height": 2200,
        "width": 1800,
        "showlegend": True,
        "template": "plotly_white",
        "legend_orientation": "h",
        "legend_yanchor": "bottom",
        "legend_y": -0.12,
        "legend_xanchor": "center",
        "legend_x": 0.5,
        "legend_title_text": "Interpolation Methods",
        "legend_title_font_size": 14,
        "margin_t": 100,
        "margin_b": 100,
        "margin_l": 50,
        "margin_r": 50,
        "font_family": "Arial",
        "font_size": 12,
    }

    _DESCRIPTION_POSITIONS = {
        "row": {1: 0.92, 2: 0.72, 3: 0.52, 4: 0.32, 5: 0.12},
        "col": {1: 0.15, 2: 0.5, 3: 0.85},
    }

    def __init__(self, save_path: Path | None = None):
        """
        Initializes the PlotlyParetoVisualizer.

        Args:
            save_path (Path | None): Optional path to save the generated plots.
        """
        super().__init__(save_path)

    def plot(self, dto: ParetoVisualizationDTO) -> None:
        """
        Generates and displays a comprehensive Pareto optimization dashboard.

        Args:
            dto (ParetoVisualizationDTO): Data Transfer Object containing all
                                          pre-processed data for visualization.
        """
        fig = self._create_figure_layout()

        # Add different sections of the visualization
        self._add_core_visualizations(fig, dto)
        self._add_parallel_coordinates(fig, dto)
        self._add_normalized_spaces(fig, dto)
        self._add_interpolation_visualizations(fig, dto)
        self._add_3d_visualizations(fig, dto)

        fig.show()
        # TODO: Add logic to save the figure if self.save_path is not None

    def _create_figure_layout(self) -> go.Figure:
        """
        Creates and configures the main figure layout for the dashboard.

        Returns:
            go.Figure: The configured Plotly figure object.
        """
        fig = make_subplots(**self._SUBPLOT_LAYOUT)

        fig.update_layout(
            title=dict(
                text=self._FIGURE_LAYOUT_CONFIG["title_text"],
                x=self._FIGURE_LAYOUT_CONFIG["title_x"],
                font=dict(size=self._FIGURE_LAYOUT_CONFIG["title_font_size"]),
            ),
            height=self._FIGURE_LAYOUT_CONFIG["height"],
            width=self._FIGURE_LAYOUT_CONFIG["width"],
            showlegend=self._FIGURE_LAYOUT_CONFIG["showlegend"],
            template=self._FIGURE_LAYOUT_CONFIG["template"],
            legend=dict(
                orientation=self._FIGURE_LAYOUT_CONFIG["legend_orientation"],
                yanchor=self._FIGURE_LAYOUT_CONFIG["legend_yanchor"],
                y=self._FIGURE_LAYOUT_CONFIG["legend_y"],
                xanchor=self._FIGURE_LAYOUT_CONFIG["legend_xanchor"],
                x=self._FIGURE_LAYOUT_CONFIG["legend_x"],
                title=dict(
                    text=self._FIGURE_LAYOUT_CONFIG["legend_title_text"],
                    font=dict(
                        size=self._FIGURE_LAYOUT_CONFIG["legend_title_font_size"]
                    ),
                ),
            ),
            margin=dict(
                t=self._FIGURE_LAYOUT_CONFIG["margin_t"],
                b=self._FIGURE_LAYOUT_CONFIG["margin_b"],
                l=self._FIGURE_LAYOUT_CONFIG["margin_l"],
                r=self._FIGURE_LAYOUT_CONFIG["margin_r"],
            ),
            font=dict(
                family=self._FIGURE_LAYOUT_CONFIG["font_family"],
                size=self._FIGURE_LAYOUT_CONFIG["font_size"],
            ),
        )
        return fig

    def _add_core_visualizations(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Adds the core decision space and objective space scatter plots.

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            dto (ParetoVisualizationDTO): The DTO containing the data.
        """
        # Decision space
        self._add_scatter_plot(
            fig,
            x=dto.pareto_set[:, 0],
            y=dto.pareto_set[:, 1],
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
            x=dto.pareto_front[:, 0],
            y=dto.pareto_front[:, 1],
            row=1,
            col=2,
            name="Pareto Front",
            color="#3498db",
            title_x="$f_1$",
            title_y="$f_2$",
            description="Objective space visualization of Pareto optimal solutions",
        )

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
        """
        Helper method to add standardized scatter plots to the figure.

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            x (np.ndarray): X-axis data.
            y (np.ndarray): Y-axis data.
            row (int): Subplot row.
            col (int): Subplot column.
            name (str): Trace name for legend.
            color (str): Marker color.
            title_x (str): X-axis title.
            title_y (str): Y-axis title.
            description (str): Text description for the subplot.
            marker_size (int): Size of the markers.
            symbol (str): Marker symbol.
            showlegend (bool): Whether to show this trace in the legend.
        """
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

    def _add_normalized_spaces(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Adds normalized decision and objective space scatter plots.

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            dto (ParetoVisualizationDTO): The DTO containing the data.
        """
        norm_x1, norm_x2 = dto.normalized_decision_space
        self._add_scatter_plot(
            fig,
            x=norm_x1,
            y=norm_x2,
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

        norm_f1, norm_f2 = dto.normalized_objective_space
        self._add_scatter_plot(
            fig,
            x=norm_f1,
            y=norm_f2,
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

    def _add_parallel_coordinates(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Adds a parallel coordinates plot for multivariate analysis.

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            dto (ParetoVisualizationDTO): The DTO containing the data.
        """
        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=dto.parallel_coordinates_data[:, 2],  # Using f1 for coloring
                    colorscale="Viridis",
                    showscale=False,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(
                        title="f₁", thickness=15, len=0.5, yanchor="middle", y=0.5
                    ),
                ),
                dimensions=[
                    dict(label="x₁", values=dto.parallel_coordinates_data[:, 0]),
                    dict(label="x₂", values=dto.parallel_coordinates_data[:, 1]),
                    dict(label="f₁", values=dto.parallel_coordinates_data[:, 2]),
                    dict(label="f₂", values=dto.parallel_coordinates_data[:, 3]),
                ],
            ),
            row=1,
            col=3,
        )
        self._add_description(fig, 1, 3, "Multivariate analysis across all dimensions")

    def _add_interpolation_visualizations(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Adds all interpolation-related visualizations.

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            dto (ParetoVisualizationDTO): The DTO containing the data.
        """
        self._add_x1_x2_interpolation(fig, dto)
        self._add_f_relationships(
            fig,
            dto.f1_relationships,
            dto.normalized_decision_space,
            dto.normalized_objective_space,
            "f1",
        )
        self._add_f_relationships(
            fig,
            dto.f2_relationships,
            dto.normalized_decision_space,
            dto.normalized_objective_space,
            "f2",
        )

    def _add_x1_x2_interpolation(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Visualizes interpolation between x1 and x2.

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            dto (ParetoVisualizationDTO): The DTO containing the data.
        """
        x1, x2 = (
            dto.x1_x2_relationship["x1"],
            dto.x1_x2_relationship["x2"],
        )
        interpolations = dto.x1_x2_relationship["interpolations"]

        self._add_scatter_plot(
            fig,
            x=x1,
            y=x2,
            row=2,
            col=3,
            name="Data Points",
            color="#3498db",
            title_x="Normalized $x_1$",
            title_y="Normalized $x_2$",
            description="Relationship between decision variables",
            showlegend=False,
        )

        for method_name, (x_grid, y_grid) in interpolations.items():
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    line=dict(
                        color=self._INTERPOLATION_COLORS[method_name],
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

    def _add_f_relationships(
        self,
        fig: go.Figure,
        f_relationships_data: dict,
        normalized_decision_space: tuple[np.ndarray, np.ndarray],
        normalized_objective_space: tuple[np.ndarray, np.ndarray],
        f_label_prefix: str,  # "f1" or "f2"
    ) -> None:
        """
        Helper to visualize relationships for a given objective function (f1 or f2).

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            f_relationships_data (dict): The dictionary from DTO (e.g., dto.f1_relationships).
            normalized_decision_space (tuple[np.ndarray, np.ndarray]): Normalized x1 and x2.
            normalized_objective_space (tuple[np.ndarray, np.ndarray]): Normalized f1 and f2.
            f_label_prefix (str): "f1" or "f2" to denote which objective is being plotted against.
        """
        norm_f_primary = f_relationships_data[f"norm_{f_label_prefix}"]
        norm_x1, norm_x2 = normalized_decision_space
        norm_f1, norm_f2 = normalized_objective_space

        # Determine target objective and its label based on f_label_prefix
        if f_label_prefix == "f1":
            norm_f_target = norm_f2
            target_f_label_str = "$f_2$"
            dto_key_f_vs_f = "f1_vs_f2"  # Key as it appears in the DTO
        else:  # f_label_prefix == "f2"
            norm_f_target = norm_f1  # Although f2_relationships in DTO doesn't currently have f2_vs_f1, keeping consistent.
            target_f_label_str = "$f_1$"
            dto_key_f_vs_f = (
                "f2_vs_f1"  # Key as it would appear in the DTO if provided.
            )

        # Base row for plotting f1 or f2 relationships
        base_row = 3 if f_label_prefix == "f1" else 4

        # Dynamically set columns based on f_label_prefix and subplot titles
        # f1 relationships will use (row 3, col 1, 2, 3)
        # f2 relationships will use (row 4, col 1, 2) (as col 3 is None)
        col_f_vs_f = 1
        col_f_vs_x1 = (
            2 if f_label_prefix == "f1" else 1
        )  # f1_vs_x1 -> (3,2); f2_vs_x1 -> (4,1)
        col_f_vs_x2 = (
            3 if f_label_prefix == "f1" else 2
        )  # f1_vs_x2 -> (3,3); f2_vs_x2 -> (4,2)

        # f_primary vs f_target (e.g., f1 vs f2, or f2 vs f1 if available)
        if (
            dto_key_f_vs_f in f_relationships_data
        ):  # Only plot if the data exists in DTO
            self._add_relationship_plot(
                fig,
                x_data=norm_f_primary,
                y_data=norm_f_target,
                interpolations=f_relationships_data[dto_key_f_vs_f],
                row=base_row,
                col=col_f_vs_f,
                x_label_title=f"Normalized ${f_label_prefix}$",
                y_label_title=f"Normalized {target_f_label_str}",
                color="#3498db",
                description=f"Relationship between ${f_label_prefix}$ and {target_f_label_str}",
            )

        # f_primary vs x1
        self._add_relationship_plot(
            fig,
            x_data=norm_f_primary,
            y_data=norm_x1,
            interpolations=f_relationships_data[f"{f_label_prefix}_vs_x1"],
            row=base_row,
            col=col_f_vs_x1,
            x_label_title=f"Normalized ${f_label_prefix}$",
            y_label_title="Normalized $x_1$",
            color="#3498db",
            description=f"Relationship between ${f_label_prefix}$ and $x_1$",
        )

        # f_primary vs x2
        self._add_relationship_plot(
            fig,
            x_data=norm_f_primary,
            y_data=norm_x2,
            interpolations=f_relationships_data[f"{f_label_prefix}_vs_x2"],
            row=base_row,
            col=col_f_vs_x2,
            x_label_title=f"Normalized ${f_label_prefix}$",
            y_label_title="Normalized $x_2$",
            color="#3498db",
            description=f"Relationship between ${f_label_prefix}$ and $x_2$",
        )

    def _add_relationship_plot(
        self,
        fig: go.Figure,
        x_data: np.ndarray,
        y_data: np.ndarray,
        interpolations: dict,
        row: int,
        col: int,
        x_label_title: str,
        y_label_title: str,
        color: str,
        description: str,
    ) -> None:
        """
        Helper method to add standardized relationship plots (scatter with interpolations).

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            x_data (np.ndarray): X-axis data for scatter points.
            y_data (np.ndarray): Y-axis data for scatter points.
            interpolations (dict): Dictionary of interpolation methods and their (x_grid, y_grid) data.
            row (int): Subplot row.
            col (int): Subplot column.
            x_label_title (str): X-axis title for the subplot.
            y_label_title (str): Y-axis title for the subplot.
            color (str): Marker color for data points.
            description (str): Text description for the subplot.
        """
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                marker=dict(size=6, opacity=0.7, color=color),
                showlegend=False,  # Data points often don't need their own legend entry here
            ),
            row=row,
            col=col,
        )

        for method_name, (x_grid, y_grid) in interpolations.items():
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    line=dict(
                        color=self._INTERPOLATION_COLORS[method_name],
                        width=2.5,
                        dash="solid" if method_name == "Pchip" else "dash",
                    ),
                    name=method_name,
                    legendgroup=method_name,  # Group for single legend entry across subplots
                    showlegend=False,  # Only show legend in the main legend area
                ),
                row=row,
                col=col,
            )

        self._set_axis_limits(
            fig, row, col, x_data, y_data, padding=0.01
        )  # Changed from [0,1] to actual data range for better fit
        fig.update_xaxes(title_text=x_label_title, row=row, col=col)
        fig.update_yaxes(title_text=y_label_title, row=row, col=col)
        self._add_description(fig, row, col, description)

    def _add_3d_visualizations(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Adds 3D visualizations of relationships using DTO data.

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            dto (ParetoVisualizationDTO): The DTO containing the data.
        """
        mv_data = dto.multivariate_interpolations
        norm_f1, norm_f2 = dto.normalized_objective_space
        norm_x1, norm_x2 = dto.normalized_decision_space

        # Track which legend items we've added in 3D plots to avoid duplication
        added_legend_items_3d = set()

        # f1, f2, x1
        self._add_3d_relationship(
            fig,
            surface_data=mv_data["f1f2_vs_x1"],
            x=norm_f1,
            y=norm_f2,
            z=norm_x1,
            row=5,
            col=1,
            z_title="x_1",
            description="3D: $f_1$, $f_2$ and $x_1$ with interpolation",
            added_legend_items=added_legend_items_3d,
        )

        # f1, f2, x2
        self._add_3d_relationship(
            fig,
            surface_data=mv_data["f1f2_vs_x2"],
            x=norm_f1,
            y=norm_f2,
            z=norm_x2,
            row=5,
            col=2,
            z_title="x_2",
            description="3D: $f_1$, $f_2$ and $x_2$ with interpolation",
            added_legend_items=added_legend_items_3d,
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
        added_legend_items: set,
    ) -> None:
        """
        Adds a 3D surface plot for multivariate relationships with original points.

        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            surface_data (dict): Dictionary of surface interpolation data (X_grid, Y_grid, Z_grid).
            x (np.ndarray): X-axis data for original points.
            y (np.ndarray): Y-axis data for original points.
            z (np.ndarray): Z-axis data for original points.
            row (int): Subplot row.
            col (int): Subplot column.
            z_title (str): Z-axis title.
            description (str): Text description for the subplot.
            added_legend_items (set): Set to track legend items already added for 3D plots
                                      to prevent duplicates in the main legend.
        """
        show_legend_points = "Original Points (3D)" not in added_legend_items
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=5, opacity=0.8, color="#1f77b4"),
                name="Original Points",
                legendgroup="original_3d_points",
                showlegend=show_legend_points,
            ),
            row=row,
            col=col,
        )
        if show_legend_points:
            added_legend_items.add("Original Points (3D)")

        for method_name, (X_grid, Y_grid, Z_grid) in surface_data.items():
            # Only create legend item once per method name for 3D surfaces
            show_in_legend = method_name not in added_legend_items

            fig.add_trace(
                go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale="Viridis",
                    opacity=0.7,
                    name=method_name,
                    showlegend=show_in_legend,
                    showscale=False,
                    colorbar=dict(
                        title=f"Norm {z_title}",
                        thickness=15,
                        len=0.5,
                        x=1.05 + 0.5 * (col - 1),
                        y=0.5,
                    ),  # Adjusted position
                    legendgroup=method_name,
                ),
                row=row,
                col=col,
            )
            if show_in_legend:
                added_legend_items.add(method_name)

        fig.update_scenes(
            row=row,
            col=col,
            xaxis_title_text="f₁",
            yaxis_title_text="f₂",
            zaxis_title_text=f"{z_title.replace('_', ' ')}",
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
        """
        Sets axis limits with padding based on the data range.

        Args:
            fig (go.Figure): The Plotly figure.
            row (int): Subplot row.
            col (int): Subplot column.
            x_data (np.ndarray): Data used to determine x-axis range.
            y_data (np.ndarray): Data used to determine y-axis range.
            padding (float): Percentage of data range to add as padding.
        """
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)

        # Handle cases where min == max (e.g., all points are the same)
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

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
        """
        Adds a description text annotation below a subplot.

        Args:
            fig (go.Figure): The Plotly figure.
            row (int): Subplot row.
            col (int): Subplot column.
            text (str): The description text to add.
        """
        row_pos = self._DESCRIPTION_POSITIONS["row"].get(row, 0)
        col_pos = self._DESCRIPTION_POSITIONS["col"].get(col, 0)

        fig.add_annotation(
            text=f"<i>{text}</i>",
            x=col_pos,
            y=row_pos,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=11, color="#7f8c8d"),
            align="center",
        )
