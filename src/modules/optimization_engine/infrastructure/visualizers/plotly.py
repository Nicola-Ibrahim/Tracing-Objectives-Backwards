from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assuming these imports are correct based on your project structure
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

    # --- Configuration Constants (now fully integrated or removed if not needed) ---
    # _INTERPOLATION_COLORS is moved into _SUBPLOT_CONFIG where needed.
    # If you still want a global constant for other uses, keep it, but it's
    # not strictly needed by the visualizer if embedded in subplots.

    _INTERPOLATION_COLORS_MAP = {  # Kept for easy reference within the config below
        "Pchip": "#07FF03",
        "Cubic Spline": "#CD05F9",
        "Linear": "#43A047",
        "Quadratic": "#7B1FA2",
        "RBF": "#F30B0B",
        "Nearest Neighbor": "#F57C00",
        "Linear ND": "#FFEA07",
    }

    # Define subplot configuration in a more structured way
    # (row, col): {type, title, description, x_label, y_label, z_label (for 3D), data_key,
    #              data_mapping: {x: 'dto_attr_for_x', y: 'dto_attr_for_y', z: 'dto_attr_for_z', ...},
    #              interpolation_key: 'dto_attr_for_interpolations' (optional for scatter)
    #              surface_key: 'dto_attr_for_surface_data' (optional for 3D)
    #              interpolation_colors: dict (new field for relevant plots)
    #             }
    _SUBPLOT_CONFIG = {
        (1, 1): {
            "type": "scatter",
            "title": "Decision Space ($x_1$ vs $x_2$)",
            "description": "Original decision variables showing trade-offs",
            "x_label": "$x_1$",
            "y_label": "$x_2$",
            "data_key": "pareto_set",
            "data_mapping": {"x": 0, "y": 1},  # Index in the numpy array
            "name": "Pareto Set",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 7,
            "showlegend": True,
        },
        (1, 2): {
            "type": "scatter",
            "title": "Objective Space ($f_1$ vs $f_2$)",
            "description": "Objective space visualization of Pareto optimal solutions",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
            "data_key": "pareto_front",
            "data_mapping": {"x": 0, "y": 1},
            "name": "Pareto Front",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 7,
            "showlegend": True,
        },
        (1, 3): {
            "type": "parcoords",
            "title": "Parallel Coordinates",
            "description": "Multivariate analysis across all dimensions",
            "data_key": "parallel_coordinates_data",
            "dimensions_labels": [
                "x₁",
                "x₂",
                "f₁",
                "f₂",
            ],  # Order corresponds to data_key columns
        },
        (2, 1): {
            "type": "scatter",
            "title": "Normalized Decision Space",
            "description": "Decision variables scaled to [0,1]",
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
            "data_key": "normalized_decision_space",  # This data_key points to tuple of arrays
            "data_mapping": {"x": 0, "y": 1},  # Use index as before for tuples (x1, x2)
            "name": "Norm Pareto Set",
            "color": "#3498db",
            "symbol": "diamond",
            "marker_size": 6,
            "showlegend": True,
        },
        (2, 2): {
            "type": "scatter",
            "title": "Normalized Objective Space",
            "description": "Objective functions normalized for comparison",
            "x_label": "Norm $f_1$",
            "y_label": "Norm $f_2$",
            "data_key": "normalized_objective_space",  # This data_key points to tuple of arrays
            "data_mapping": {"x": 0, "y": 1},  # Use index as before for tuples (f1, f2)
            "name": "Norm Pareto Front",
            "color": "#3498db",
            "symbol": "diamond",
            "marker_size": 6,
            "showlegend": True,
        },
        (2, 3): {
            "type": "scatter",
            "title": "$x_1$ vs $x_2$ (Interpolations)",
            "description": "Relationship between decision variables",
            "x_label": "Normalized $x_1$",
            "y_label": "Normalized $x_2$",
            "data_key": "x1_x2_relationship",  # The relationship dict now includes 'x1' and 'x2' directly
            "data_mapping": {
                "x": "x1",
                "y": "x2",
            },  # Key in the x1_x2_relationship dict
            "interpolation_key": "interpolations",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
            "interpolation_colors": _INTERPOLATION_COLORS_MAP,  # Embed colors
        },
        (3, 1): {
            "type": "scatter",
            "title": "$f_1$ vs $f_2$ (Interpolations)",
            "description": "Relationship between $f_1$ and $f_2$",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $f_2$",
            "data_key": "f1_relationships",
            "data_mapping": {
                "x": "norm_f1",
                "y": "norm_f2",
            },
            "interpolation_key": "f1_vs_f2",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
            "interpolation_colors": _INTERPOLATION_COLORS_MAP,
        },
        (3, 2): {
            "type": "scatter",
            "title": "$f_1$ vs $x_1$ (Interpolations)",
            "description": "Relationship between $f_1$ and $x_1$",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $x_1$",
            "data_key": "f1_relationships",
            "data_mapping": {
                "x": "norm_f1",
                "y": "norm_x1",
            },
            "interpolation_key": "f1_vs_x1",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
            "interpolation_colors": _INTERPOLATION_COLORS_MAP,
        },
        (3, 3): {
            "type": "scatter",
            "title": "$f_1$ vs $x_2$ (Interpolations)",
            "description": "Relationship between $f_1$ and $x_2$",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $x_2$",
            "data_key": "f1_relationships",
            "data_mapping": {
                "x": "norm_f1",
                "y": "norm_x2",
            },
            "interpolation_key": "f1_vs_x2",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
            "interpolation_colors": _INTERPOLATION_COLORS_MAP,
        },
        (4, 1): {
            "type": "scatter",
            "title": "$f_2$ vs $x_1$ (Interpolations)",
            "description": "Relationship between $f_2$ and $x_1$",
            "x_label": "Normalized $f_2$",
            "y_label": "Normalized $x_1$",
            "data_key": "f2_relationships",
            "data_mapping": {
                "x": "norm_f2",
                "y": "norm_x1",
            },
            "interpolation_key": "f2_vs_x1",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
            "interpolation_colors": _INTERPOLATION_COLORS_MAP,
        },
        (4, 2): {
            "type": "scatter",
            "title": "$f_2$ vs $x_2$ (Interpolations)",
            "description": "Relationship between $f_2$ and $x_2$",
            "x_label": "Normalized $f_2$",
            "y_label": "Normalized $x_2$",
            "data_key": "f2_relationships",
            "data_mapping": {
                "x": "norm_f2",
                "y": "norm_x2",
            },
            "interpolation_key": "f2_vs_x2",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
            "interpolation_colors": _INTERPOLATION_COLORS_MAP,
        },
        (5, 1): {
            "type": "scatter3d",
            "title": "3D: $f_1$, $f_2$, $x_1$",
            "description": "3D: $f_1$, $f_2$ and $x_1$ with interpolation",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
            "z_label": "$x_1$",
            "data_key": "norm_f1",  # Directly use norm_f1 from DTO
            "data_mapping": {"x": None},  # Data is already x, no further mapping needed
            "y_data_key": "norm_f2",  # New: direct key for y-axis
            "z_data_key": "norm_x1",  # Direct key for z-axis
            "surface_key": "f1f2_vs_x1",
            "name": "Original Points",
            "color": "#1f77b4",
            "marker_size": 5,
            "interpolation_colors": _INTERPOLATION_COLORS_MAP,
        },
        (5, 2): {
            "type": "scatter3d",
            "title": "3D: $f_1$, $f_2$, $x_2$",
            "description": "3D: $f_1$, $f_2$ and $x_2$ with interpolation",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
            "z_label": "$x_2$",
            "data_key": "norm_f1",  # Directly use norm_f1 from DTO
            "data_mapping": {"x": None},  # Data is already x, no further mapping needed
            "y_data_key": "norm_f2",
            "z_data_key": "norm_x2",
            "surface_key": "f1f2_vs_x2",
            "name": "Original Points",
            "color": "#1f77b4",
            "marker_size": 5,
            "interpolation_colors": _INTERPOLATION_COLORS_MAP,
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
        self._added_legend_items = set()

    def plot(self, dto: ParetoVisualizationDTO) -> None:
        """
        Generates and displays a comprehensive Pareto optimization dashboard.

        Args:
            dto (ParetoVisualizationDTO): Data Transfer Object containing all
                                          pre-processed data for visualization.
        """
        fig = self._create_figure_layout()
        self._add_all_subplots_from_config(fig, dto)

        fig.show()

    def _create_figure_layout(self) -> go.Figure:
        """
        Creates and configures the main figure layout for the dashboard.

        Returns:
            go.Figure: The configured Plotly figure object.
        """

        rows = max(r for r, c in self._SUBPLOT_CONFIG.keys())
        cols = max(c for r, c in self._SUBPLOT_CONFIG.keys())

        specs = [[None for _ in range(cols)] for _ in range(rows)]
        subplot_titles = [None] * (rows * cols)

        for (r, c), config in self._SUBPLOT_CONFIG.items():
            if 1 <= r <= rows and 1 <= c <= cols:
                specs[r - 1][c - 1] = {"type": config["type"]}
                subplot_titles[(r - 1) * cols + (c - 1)] = config["title"]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.1,
            column_widths=[
                0.3,
                0.3,
                0.4,
            ],
            row_heights=[0.15, 0.15, 0.15, 0.15, 0.4],
        )

        fig.update_layout(
            title=dict(
                text=self._FIGURE_LAYOUT_CONFIG["title_text"],
                x=self._FIGURE_LAYOUT_CONFIG["title_x"],
                font=dict(size=self._FIGURE_LAYOUT_CONFIG["title_font_size"]),
            ),
            height=self._FIGURE_LAYOUT_CONFIG["height"],
            width=self._FIGURE_LAYOUT_CONFIG["width"],  # Fixed typo
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

    def _add_all_subplots_from_config(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Iterates through _SUBPLOT_CONFIG and adds traces dynamically.
        """
        for (row, col), config in self._SUBPLOT_CONFIG.items():
            plot_type = config.get("type")
            description = config.get("description", "")

            # For scatter and scatter3d, data_key and data_mapping now work differently
            # We now primarily access data directly via DTO attributes.

            if plot_type == "scatter":
                # For scatter plots, data_key might point to a dictionary (e.g., f1_relationships)
                # or a tuple/array (e.g., pareto_set, normalized_decision_space)
                data_container = getattr(
                    dto, config.get("data_key")
                )  # Get the dict or tuple

                x_data = self._get_data_from_mapping(
                    data_container, config.get("data_mapping", {}).get("x")
                )
                y_data = self._get_data_from_mapping(
                    data_container, config.get("data_mapping", {}).get("y")
                )

                if (
                    x_data is None
                    or y_data is None
                    or x_data.size == 0
                    or y_data.size == 0
                ):
                    print(
                        f"Warning: Insufficient x or y data for scatter plot at ({row}, {col}). Skipping."
                    )
                    continue

                self._add_scatter_plot(fig, row, col, x_data, y_data, config)

                # Add interpolations if configured
                if "interpolation_key" in config:
                    interpolations = self._get_data_from_mapping(
                        data_container, config["interpolation_key"]
                    )
                    if interpolations:
                        self._add_interpolation_traces(
                            fig,
                            row,
                            col,
                            interpolations,
                            config.get("interpolation_colors", {}),
                        )

            elif plot_type == "parcoords":
                parcoords_data = getattr(dto, config.get("data_key"), None)
                if parcoords_data is None:
                    print(
                        f"Warning: Missing data for parcoords plot at ({row}, {col}). Skipping."
                    )
                    continue
                self._add_parcoords_plot(fig, row, col, parcoords_data, config)

            elif plot_type == "scatter3d":
                x_data = getattr(
                    dto, config.get("data_key"), None
                )  # Direct access to norm_f1/f2/x1/x2
                y_data = getattr(
                    dto, config.get("y_data_key"), None
                )  # Direct access to norm_f1/f2/x1/x2
                z_data = getattr(
                    dto, config.get("z_data_key"), None
                )  # Direct access to norm_f1/f2/x1/x2

                if (
                    x_data is None
                    or y_data is None
                    or z_data is None
                    or x_data.size == 0
                    or y_data.size == 0
                    or z_data.size == 0
                ):
                    print(
                        f"Warning: Insufficient x, y, or z data for 3D scatter plot at ({row}, {col}). Skipping."
                    )
                    continue

                self._add_scatter3d_plot(fig, row, col, x_data, y_data, z_data, config)

                # Add surface interpolations if configured
                if "surface_key" in config:
                    surface_data = getattr(dto, "multivariate_interpolations", {}).get(
                        config["surface_key"]
                    )
                    if surface_data:
                        self._add_3d_surface_traces(
                            fig,
                            row,
                            col,
                            surface_data,
                            config.get("interpolation_colors", {}),
                        )
            else:
                print(
                    f"Warning: Unsupported plot type '{plot_type}' for subplot ({row}, {col})."
                )
                continue

            self._add_description(fig, row, col, description)

    def _get_data_from_mapping(self, data_source, mapping_key) -> np.ndarray | None:
        """
        Helper to retrieve data from a given data_source based on the mapping key.
        Mapping key can be an integer (for numpy array index) or a string (for dict key).
        Lambda functions are no longer expected here.
        """
        if (
            data_source is None or mapping_key is None
        ):  # Added check for mapping_key being None
            return data_source  # If data_key directly holds the array and mapping_key is None (e.g. for 3D x-data)

        if isinstance(mapping_key, int):
            if isinstance(data_source, (np.ndarray, list, tuple)):
                try:
                    if isinstance(data_source, np.ndarray) and data_source.ndim > 1:
                        return data_source[:, mapping_key]
                    else:
                        return np.asarray(data_source[mapping_key])
                except (IndexError, TypeError):
                    print(
                        f"Error: Invalid index {mapping_key} for data source type {type(data_source)}."
                    )
                    return None
            else:
                print(
                    f"Error: Integer index {mapping_key} not supported for data source type {type(data_source)}."
                )
                return None
        elif isinstance(mapping_key, str):
            if isinstance(data_source, dict):
                return data_source.get(mapping_key)
            else:
                print(
                    f"Error: String key '{mapping_key}' not supported for data source type {type(data_source)} (expected dict)."
                )
                return None
        # No more callable (lambda) handling here
        return None

    def _add_scatter_plot(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        config: dict,
    ) -> None:
        """
        Adds a single scatter plot based on config.
        """
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=config.get("marker_size", 7),
                    opacity=0.8,
                    color=config.get("color", "#3498db"),
                    symbol=config.get("symbol", "circle"),
                ),
                name=config.get("name", "Data Points"),
                showlegend=config.get("showlegend", True),
            ),
            row=row,
            col=col,
        )
        self._set_axis_limits(fig, row, col, x, y)
        fig.update_xaxes(title_text=config["x_label"], row=row, col=col)
        fig.update_yaxes(title_text=config["y_label"], row=row, col=col)

    def _add_parcoords_plot(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        data: np.ndarray,
        config: dict,
    ) -> None:
        """
        Adds a parallel coordinates plot based on config.
        """
        dimensions_labels = config.get("dimensions_labels", [])
        dimensions = []
        for i, label in enumerate(dimensions_labels):
            if i < data.shape[1]:
                dimensions.append(dict(label=label, values=data[:, i]))
            else:
                print(
                    f"Warning: Label '{label}' specified for dimension {i} but data has only {data.shape[1]} columns."
                )

        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=data[:, 2] if data.shape[1] > 2 else data[:, 0],
                    colorscale="Viridis",
                    showscale=False,
                    cmin=np.min(data[:, 2])
                    if data.shape[1] > 2
                    else np.min(data[:, 0]),
                    cmax=np.max(data[:, 2])
                    if data.shape[1] > 2
                    else np.max(data[:, 0]),
                ),
                dimensions=dimensions,
            ),
            row=row,
            col=col,
        )

    def _add_interpolation_traces(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        interpolations: dict,
        interpolation_colors: dict,  # Now passed directly from config
    ) -> None:
        """
        Adds interpolation lines for a scatter plot.
        """
        for method_name, (x_grid, y_grid) in interpolations.items():
            show_legend_item = method_name not in self._added_legend_items
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    line=dict(
                        color=interpolation_colors.get(method_name, "#000000"),
                        width=2.5,
                        dash="solid" if method_name == "Pchip" else "dash",
                    ),
                    name=method_name,
                    legendgroup=method_name,
                    showlegend=show_legend_item,
                ),
                row=row,
                col=col,
            )
            if show_legend_item:
                self._added_legend_items.add(method_name)

    def _add_scatter3d_plot(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        config: dict,
    ) -> None:
        """
        Adds a 3D scatter plot.
        """
        show_legend_points = (
            config.get("name", "Original Points") not in self._added_legend_items
        )

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=config.get("marker_size", 5),
                    opacity=0.8,
                    color=config.get("color", "#1f77b4"),
                ),
                name=config.get("name", "Original Points"),
                legendgroup="original_3d_points",
                showlegend=show_legend_points,
            ),
            row=row,
            col=col,
        )
        if show_legend_points:
            self._added_legend_items.add(config.get("name", "Original Points"))

        fig.update_scenes(
            xaxis_title_text=config["x_label"],
            yaxis_title_text=config["y_label"],
            zaxis_title_text=config["z_label"],
            aspectmode="cube",
            row=row,
            col=col,
        )

    def _add_3d_surface_traces(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        surface_data: dict,
        interpolation_colors: dict,  # Now passed directly from config
    ) -> None:
        """
        Adds 3D surface interpolations.
        """
        for method_name, (X_grid, Y_grid, Z_grid) in surface_data.items():
            show_in_legend = method_name not in self._added_legend_items

            fig.add_trace(
                go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale=[
                        [0, interpolation_colors.get(method_name, "#000000")],
                        [1, interpolation_colors.get(method_name, "#000000")],
                    ],
                    opacity=0.7,
                    name=method_name,
                    showlegend=show_in_legend,
                    showscale=False,
                    legendgroup=method_name,
                ),
                row=row,
                col=col,
            )
            if show_in_legend:
                self._added_legend_items.add(method_name)

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
        if x_data.size == 0 or y_data.size == 0:
            return

        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)

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
