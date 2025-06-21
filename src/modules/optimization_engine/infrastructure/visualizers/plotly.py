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

    # Define subplot configuration in a more structured way
    # (row, col): {type, title, description, x_label, y_label, z_label (for 3D)}
    _SUBPLOT_CONFIG = {
        (1, 1): {
            "type": "scatter",
            "title": "Decision Space ($x_1$ vs $x_2$)",
            "description": "Original decision variables showing trade-offs",
            "x_label": "$x_1$",
            "y_label": "$x_2$",
        },
        (1, 2): {
            "type": "scatter",
            "title": "Objective Space ($f_1$ vs $f_2$)",
            "description": "Objective space visualization of Pareto optimal solutions",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
        },
        (1, 3): {
            "type": "parcoords",
            "title": "Parallel Coordinates",
            "description": "Multivariate analysis across all dimensions",
        },
        (2, 1): {
            "type": "scatter",
            "title": "Normalized Decision Space",
            "description": "Decision variables scaled to [0,1]",
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
        },
        (2, 2): {
            "type": "scatter",
            "title": "Normalized Objective Space",
            "description": "Objective functions normalized for comparison",
            "x_label": "Norm $f_1$",
            "y_label": "Norm $f_2$",
        },
        (2, 3): {
            "type": "scatter",
            "title": "$x_1$ vs $x_2$ (Interpolations)",
            "description": "Relationship between decision variables",
            "x_label": "Normalized $x_1$",
            "y_label": "Normalized $x_2$",
        },
        (3, 1): {
            "type": "scatter",
            "title": "$f_1$ vs $f_2$ (Interpolations)",
            "description": "Relationship between $f_1$ and $f_2$",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $f_2$",
        },
        (3, 2): {
            "type": "scatter",
            "title": "$f_1$ vs $x_1$ (Interpolations)",
            "description": "Relationship between $f_1$ and $x_1$",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $x_1$",
        },
        (3, 3): {
            "type": "scatter",
            "title": "$f_1$ vs $x_2$ (Interpolations)",
            "description": "Relationship between $f_1$ and $x_2$",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $x_2$",
        },
        (4, 1): {
            "type": "scatter",
            "title": "$f_2$ vs $x_1$ (Interpolations)",
            "description": "Relationship between $f_2$ and $x_1$",
            "x_label": "Normalized $f_2$",
            "y_label": "Normalized $x_1$",
        },
        (4, 2): {
            "type": "scatter",
            "title": "$f_2$ vs $x_2$ (Interpolations)",
            "description": "Relationship between $f_2$ and $x_2$",
            "x_label": "Normalized $f_2$",
            "y_label": "Normalized $x_2$",
        },
        (5, 1): {
            "type": "scatter3d",
            "title": "3D: $f_1$, $f_2$, $x_1$",
            "description": "3D: $f_1$, $f_2$ and $x_1$ with interpolation",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $f_2$",
            "z_label": "Normalized $x_1$",
        },
        (5, 2): {
            "type": "scatter3d",
            "title": "3D: $f_1$, $f_2$, $x_2$",
            "description": "3D: $f_1$, $f_2$ and $x_2$ with interpolation",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $f_2$",
            "z_label": "Normalized $x_2$",
        },
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
        # Set to keep track of added legend items to avoid duplicates in 3D plots
        self._added_3d_legend_items = set()

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
        if self.save_path:
            # Ensure the directory exists
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(self.save_path))
            print(f"Dashboard saved to: {self.save_path}")

    def _create_figure_layout(self) -> go.Figure:
        """
        Creates and configures the main figure layout for the dashboard.

        Returns:
            go.Figure: The configured Plotly figure object.
        """
        # Dynamically create specs and subplot_titles from _SUBPLOT_CONFIG
        rows = max(r for r, c in self._SUBPLOT_CONFIG.keys())
        cols = max(c for r, c in self._SUBPLOT_CONFIG.keys())

        # Initialize specs and titles with None for all potential cells
        specs = [[None for _ in range(cols)] for _ in range(rows)]
        subplot_titles = [None] * (rows * cols)

        for (r, c), config in self._SUBPLOT_CONFIG.items():
            if 1 <= r <= rows and 1 <= c <= cols:  # Ensure indices are within bounds
                specs[r - 1][c - 1] = {"type": config["type"]}
                subplot_titles[(r - 1) * cols + (c - 1)] = config["title"]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.1,
            # These can also be made dynamic if needed, but for now fixed as per original
            column_widths=[0.3, 0.3, 0.4],
            row_heights=[0.15, 0.15, 0.15, 0.15, 0.4],
        )

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
        """
        # Decision space
        self._add_scatter_with_description(
            fig,
            x=dto.pareto_set[:, 0],
            y=dto.pareto_set[:, 1],
            row=1,
            col=1,
            name="Pareto Set",
            color="#3498db",
            marker_size=7,
            symbol="circle",
            showlegend=True,
        )

        # Objective space
        self._add_scatter_with_description(
            fig,
            x=dto.pareto_front[:, 0],
            y=dto.pareto_front[:, 1],
            row=1,
            col=2,
            name="Pareto Front",
            color="#3498db",
            marker_size=7,
            symbol="circle",
            showlegend=True,
        )

    def _add_scatter_with_description(
        self,
        fig: go.Figure,
        x: np.ndarray,
        y: np.ndarray,
        row: int,
        col: int,
        name: str,
        color: str,
        marker_size: int = 7,
        symbol: str = "circle",
        showlegend: bool = True,
    ) -> None:
        """
        Helper method to add standardized scatter plots to the figure,
        including axis titles and descriptions based on _SUBPLOT_CONFIG.
        """
        subplot_info = self._SUBPLOT_CONFIG.get((row, col))
        if not subplot_info:
            print(f"Warning: No subplot configuration found for ({row}, {col})")
            return

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
        fig.update_xaxes(title_text=subplot_info["x_label"], row=row, col=col)
        fig.update_yaxes(title_text=subplot_info["y_label"], row=row, col=col)
        self._add_description(fig, row, col, subplot_info["description"])

    def _add_normalized_spaces(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Adds normalized decision and objective space scatter plots.
        """
        norm_x1, norm_x2 = dto.normalized_decision_space
        self._add_scatter_with_description(
            fig,
            x=norm_x1,
            y=norm_x2,
            row=2,
            col=1,
            name="Norm Pareto Set",
            color="#3498db",
            marker_size=6,
            symbol="diamond",
            showlegend=True,
        )

        norm_f1, norm_f2 = dto.normalized_objective_space
        self._add_scatter_with_description(
            fig,
            x=norm_f1,
            y=norm_f2,
            row=2,
            col=2,
            name="Norm Pareto Front",
            color="#3498db",
            marker_size=6,
            symbol="diamond",
            showlegend=True,
        )

    def _add_parallel_coordinates(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Adds a parallel coordinates plot for multivariate analysis.
        """
        subplot_info = self._SUBPLOT_CONFIG.get((1, 3))
        if not subplot_info:
            print(f"Warning: No subplot configuration found for (1, 3)")
            return

        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=dto.parallel_coordinates_data[:, 2],  # Using f1 for coloring
                    colorscale="Viridis",
                    showscale=False,  # Color scale is added as a separate annotation if needed
                    cmin=np.min(dto.parallel_coordinates_data[:, 2]),
                    cmax=np.max(dto.parallel_coordinates_data[:, 2]),
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
        self._add_description(fig, 1, 3, subplot_info["description"])

    def _add_interpolation_visualizations(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Adds all interpolation-related visualizations.
        """
        self._add_x1_x2_interpolation(fig, dto)
        self._add_f_relationships(fig, dto, "f1")
        self._add_f_relationships(fig, dto, "f2")

    def _add_x1_x2_interpolation(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Visualizes interpolation between x1 and x2.
        """
        subplot_info = self._SUBPLOT_CONFIG.get((2, 3))
        if not subplot_info:
            print(f"Warning: No subplot configuration found for (2, 3)")
            return

        x1_data = dto.x1_x2_relationship.get("x1")
        x2_data = dto.x1_x2_relationship.get("x2")
        interpolations = dto.x1_x2_relationship.get("interpolations", {})

        if x1_data is None or x2_data is None:
            print("Warning: Missing x1 or x2 data for x1_x2_interpolation.")
            return

        # Add data points
        fig.add_trace(
            go.Scatter(
                x=x1_data,
                y=x2_data,
                mode="markers",
                marker=dict(size=6, opacity=0.7, color="#3498db"),
                name="Data Points",
                showlegend=False,
            ),
            row=2,
            col=3,
        )

        # Add interpolation lines
        for method_name, (x_grid, y_grid) in interpolations.items():
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    line=dict(
                        color=self._INTERPOLATION_COLORS.get(method_name, "#000000"),
                        width=3,
                        dash="solid" if method_name == "Pchip" else "dash",
                    ),
                    name=method_name,
                    legendgroup=method_name,
                    showlegend=True,  # Show legend for interpolation methods
                ),
                row=2,
                col=3,
            )

        self._set_axis_limits(fig, 2, 3, x1_data, x2_data)
        fig.update_xaxes(title_text=subplot_info["x_label"], row=2, col=3)
        fig.update_yaxes(title_text=subplot_info["y_label"], row=2, col=3)
        self._add_description(fig, 2, 3, subplot_info["description"])

    def _add_f_relationships(
        self, fig: go.Figure, dto: ParetoVisualizationDTO, f_prefix: str
    ) -> None:
        """
        Helper to visualize relationships for a given objective function (f1 or f2).
        Args:
            fig (go.Figure): The Plotly figure to add traces to.
            dto (ParetoVisualizationDTO): The DTO containing the data.
            f_prefix (str): "f1" or "f2" to denote which objective is being plotted against.
        """
        f_relationships_data = (
            dto.f1_relationships if f_prefix == "f1" else dto.f2_relationships
        )
        norm_f_primary = f_relationships_data.get(f"norm_{f_prefix}")
        if norm_f_primary is None:
            print(f"Warning: Missing normalized {f_prefix} data.")
            return

        norm_x1, norm_x2 = dto.normalized_decision_space
        norm_f1, norm_f2 = dto.normalized_objective_space

        # Determine target objective and its label based on f_prefix
        if f_prefix == "f1":
            norm_f_target = norm_f2
            target_f_label_str = "$f_2$"
            dto_key_f_vs_f = "f1_vs_f2"
        else:  # f_prefix == "f2"
            norm_f_target = norm_f1
            target_f_label_str = "$f_1$"
            dto_key_f_vs_f = "f2_vs_f1"

        # Base row for plotting f1 or f2 relationships
        base_row = 3 if f_prefix == "f1" else 4

        # f_primary vs f_target (e.g., f1 vs f2, or f2 vs f1 if available)
        # Only plot if the data exists in DTO and the corresponding subplot is defined
        if dto_key_f_vs_f in f_relationships_data and self._SUBPLOT_CONFIG.get(
            (base_row, 1)
        ):
            self._add_relationship_plot(
                fig,
                x_data=norm_f_primary,
                y_data=norm_f_target,
                interpolations=f_relationships_data[dto_key_f_vs_f],
                row=base_row,
                col=1,
            )

        # f_primary vs x1
        if f"{f_prefix}_vs_x1" in f_relationships_data:
            col_f_vs_x1 = 2 if f_prefix == "f1" else 1
            self._add_relationship_plot(
                fig,
                x_data=norm_f_primary,
                y_data=norm_x1,
                interpolations=f_relationships_data[f"{f_prefix}_vs_x1"],
                row=base_row,
                col=col_f_vs_x1,
            )

        # f_primary vs x2
        if f"{f_prefix}_vs_x2" in f_relationships_data:
            col_f_vs_x2 = 3 if f_prefix == "f1" else 2
            self._add_relationship_plot(
                fig,
                x_data=norm_f_primary,
                y_data=norm_x2,
                interpolations=f_relationships_data[f"{f_prefix}_vs_x2"],
                row=base_row,
                col=col_f_vs_x2,
            )

    def _add_relationship_plot(
        self,
        fig: go.Figure,
        x_data: np.ndarray,
        y_data: np.ndarray,
        interpolations: dict,
        row: int,
        col: int,
    ) -> None:
        """
        Helper method to add standardized relationship plots (scatter with interpolations),
        fetching titles and descriptions from _SUBPLOT_CONFIG.
        """
        subplot_info = self._SUBPLOT_CONFIG.get((row, col))
        if not subplot_info:
            print(f"Warning: No subplot configuration found for ({row}, {col})")
            return

        # Add data points
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                marker=dict(size=6, opacity=0.7, color="#3498db"),
                showlegend=False,  # Data points often don't need their own legend entry here
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
                        color=self._INTERPOLATION_COLORS.get(method_name, "#000000"),
                        width=2.5,
                        dash="solid" if method_name == "Pchip" else "dash",
                    ),
                    name=method_name,
                    legendgroup=method_name,  # Group for single legend entry across subplots
                    showlegend=False,  # Only show legend in the main legend area (handled by _add_x1_x2_interpolation for first instance)
                ),
                row=row,
                col=col,
            )

        self._set_axis_limits(fig, row, col, x_data, y_data, padding=0.01)
        fig.update_xaxes(title_text=subplot_info["x_label"], row=row, col=col)
        fig.update_yaxes(title_text=subplot_info["y_label"], row=row, col=col)
        self._add_description(fig, row, col, subplot_info["description"])

    def _add_3d_visualizations(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Adds 3D visualizations of relationships using DTO data.
        """
        mv_data = dto.multivariate_interpolations
        norm_f1, norm_f2 = dto.normalized_objective_space
        norm_x1, norm_x2 = dto.normalized_decision_space

        # f1, f2, x1
        if "f1f2_vs_x1" in mv_data:
            self._add_3d_relationship(
                fig,
                surface_data=mv_data["f1f2_vs_x1"],
                x=norm_f1,
                y=norm_f2,
                z=norm_x1,
                row=5,
                col=1,
            )

        # f1, f2, x2
        if "f1f2_vs_x2" in mv_data:
            self._add_3d_relationship(
                fig,
                surface_data=mv_data["f1f2_vs_x2"],
                x=norm_f1,
                y=norm_f2,
                z=norm_x2,
                row=5,
                col=2,
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
    ) -> None:
        """
        Adds a 3D surface plot for multivariate relationships with original points,
        fetching titles and descriptions from _SUBPLOT_CONFIG.
        """
        subplot_info = self._SUBPLOT_CONFIG.get((row, col))
        if not subplot_info:
            print(f"Warning: No subplot configuration found for ({row}, {col})")
            return

        # Add original data points
        show_legend_points = "Original Points (3D)" not in self._added_3d_legend_items
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
            self._added_3d_legend_items.add("Original Points (3D)")

        # Add surface interpolations
        for method_name, (X_grid, Y_grid, Z_grid) in surface_data.items():
            # Only create legend item once per method name for 3D surfaces
            show_in_legend = method_name not in self._added_3d_legend_items

            fig.add_trace(
                go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale="Viridis",
                    opacity=0.7,
                    name=method_name,
                    showlegend=show_in_legend,
                    showscale=False,  # Managed by shared colorbar if desired, or per-plot if needed
                    legendgroup=method_name,
                ),
                row=row,
                col=col,
            )
            if show_in_legend:
                self._added_3d_legend_items.add(method_name)

        # Update 3D scene (axis titles)
        fig.update_scenes(
            xaxis_title_text=subplot_info["x_label"],
            yaxis_title_text=subplot_info["y_label"],
            zaxis_title_text=subplot_info["z_label"],
            aspectmode="cube",
            row=row,
            col=col,
        )
        self._add_description(fig, row, col, subplot_info["description"])

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
        """
        # Ensure data is not empty to avoid errors
        if x_data.size == 0 or y_data.size == 0:
            return

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
