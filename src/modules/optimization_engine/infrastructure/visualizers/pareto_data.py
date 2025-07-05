from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import scipy.interpolate as spi
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

from ...domain.analysis.interfaces.base_visualizer import BaseDataVisualizer


@dataclass
class ParetoData:
    """
    Comprehensive container for all Pareto optimization data.
    Includes original, normalized values, and pre-computed interpolations.
    This replaces ParetoDataset and ParetoVisualizationDTO.
    """

    # Core Pareto Data (original values)
    pareto_set: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={
            "description": "Original decision variables (X) for Pareto optimal solutions. Shape (n_samples, n_decision_vars)."
        },
    )
    pareto_front: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={
            "description": "Original objective function values (F) for Pareto optimal solutions. Shape (n_samples, n_objective_vars)."
        },
    )

    # Normalized Values (0-1 range) - Directly stored as attributes
    norm_f1: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Normalized values of objective function 1."},
    )
    norm_f2: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Normalized values of objective function 2."},
    )
    norm_x1: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Normalized values of decision variable 1."},
    )
    norm_x2: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Normalized values of decision variable 2."},
    )

    # Data for parallel coordinates plot (derived from normalized data)
    parallel_coordinates_data: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={
            "description": "Combined normalized data for parallel coordinates plot."
        },
    )

    # 1D Interpolations: {relationship_name: {method_name: (x_grid, y_grid)}}
    interpolations_1d: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=lambda: {
            "f1_vs_f2": {},
            "f1_vs_x1": {},
            "f1_vs_x2": {},
            "x1_vs_x2": {},
            "f2_vs_x1": {},
            "f2_vs_x2": {},
        },
        metadata={
            "description": "1D interpolated data for various variable relationships."
        },
    )

    # 2D Multivariate Interpolations: {relationship_name: {method_name: (X_grid, Y_grid, Z_grid)}}
    interpolations_2d: dict[
        str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = field(
        default_factory=lambda: {
            "f1f2_vs_x1": {},
            "f1f2_vs_x2": {},
        },
        metadata={
            "description": "2D interpolated data for multivariate relationships (surfaces)."
        },
    )


class ParetoDataService:
    """
    Service for preparing comprehensive Pareto optimization data,
    including normalization and various interpolations.
    """

    INTERPOLATION_METHODS_1D: dict[str, Any] = {
        "Pchip": spi.PchipInterpolator,
        "Cubic Spline": spi.CubicSpline,
        "Linear": "linear",  # 'kind' for interp1d
        "Quadratic": "quadratic",  # 'kind' for interp1d
        "RBF": spi.RBFInterpolator,
    }

    INTERPOLATION_METHODS_ND: dict[str, Any] = {
        "Nearest Neighbor": spi.NearestNDInterpolator,
        "Linear ND": spi.LinearNDInterpolator,
    }

    _NUM_INTERPOLATION_POINTS_1D = 100
    _NUM_INTERPOLATION_POINTS_2D_GRID = 50

    def prepare_data(self, raw_pareto_data: Any) -> ParetoData:
        """
        Prepares and populates a ParetoData object from raw Pareto optimization results.

        Args:
            raw_pareto_data (Any): An object expected to have 'pareto_set' and 'pareto_front' attributes.

        Returns:
            ParetoData: A comprehensive data object with original, normalized,
                        and interpolated Pareto data.

        Raises:
            ValueError: If input data is invalid or insufficient.
        """
        if not hasattr(raw_pareto_data, "pareto_set") or not hasattr(
            raw_pareto_data, "pareto_front"
        ):
            raise ValueError(
                "Raw Pareto data must have 'pareto_set' and 'pareto_front' attributes."
            )

        if (
            raw_pareto_data.pareto_set.size == 0
            or raw_pareto_data.pareto_front.size == 0
        ):
            raise ValueError(
                "Loaded Pareto data (pareto_set or pareto_front) is empty."
            )

        data = ParetoData(
            pareto_set=raw_pareto_data.pareto_set,
            pareto_front=raw_pareto_data.pareto_front,
        )

        if data.pareto_front.shape[1] < 2:
            raise ValueError("pareto_front must have at least 2 columns for f1 and f2.")
        if data.pareto_set.shape[1] < 2:
            raise ValueError("pareto_set must have at least 2 columns for x1 and x2.")

        # Populate normalized attributes directly
        data.norm_f1 = self._normalize_array(data.pareto_front[:, 0])
        data.norm_f2 = self._normalize_array(data.pareto_front[:, 1])
        data.norm_x1 = self._normalize_array(data.pareto_set[:, 0])
        data.norm_x2 = self._normalize_array(data.pareto_set[:, 1])

        # Populate parallel coordinates data
        data.parallel_coordinates_data = np.hstack(
            [
                data.norm_x1.reshape(-1, 1),
                data.norm_x2.reshape(-1, 1),
                data.norm_f1.reshape(-1, 1),
                data.norm_f2.reshape(-1, 1),
            ]
        )

        self._compute_all_1d_interpolations(data)
        self._compute_all_2d_interpolations(data)

        return data

    def _normalize_array(self, data_array: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range using Min-Max scaling"""
        if data_array.size == 0:
            return np.array([])
        scaler = MinMaxScaler()
        reshaped_data = data_array.reshape(-1, 1)
        normalized_data = scaler.fit_transform(reshaped_data)
        return normalized_data.flatten()

    def _preprocess_1d_data(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sorts data by x and removes duplicate x values to ensure a strictly
        increasing sequence, as required by many SciPy interpolators.
        """
        if x.size == 0 or y.size == 0:
            return np.array([]), np.array([])

        sort_indices = np.argsort(x)
        x_sorted = x[sort_indices]
        y_sorted = y[sort_indices]

        # Use a boolean mask for unique elements for efficiency
        # This handles floats robustly.
        unique_mask = np.append(True, np.diff(x_sorted) != 0)

        return x_sorted[unique_mask], y_sorted[unique_mask]

    def _compute_all_1d_interpolations(self, data: ParetoData):
        """Fit and predict 1D scipy.interpolate mappers for various relationships."""
        # Using the direct attributes from ParetoData
        relationship_data_map = {
            "f1_vs_f2": {"x_train": data.norm_f1, "y_train": data.norm_f2},
            "f1_vs_x1": {"x_train": data.norm_f1, "y_train": data.norm_x1},
            "f1_vs_x2": {"x_train": data.norm_f1, "y_train": data.norm_x2},
            "x1_vs_x2": {"x_train": data.norm_x1, "y_train": data.norm_x2},
            "f2_vs_x1": {"x_train": data.norm_f2, "y_train": data.norm_x1},
            "f2_vs_x2": {"x_train": data.norm_f2, "y_train": data.norm_x2},
        }

        for relationship_name, data_sources in relationship_data_map.items():
            x_train = data_sources["x_train"]
            y_train = data_sources["y_train"]

            if x_train.size < 2 or y_train.size < 2:
                print(
                    f"Warning: Not enough data points to interpolate for '{relationship_name}'. Skipping."
                )
                continue

            x_train_processed, y_train_processed = self._preprocess_1d_data(
                x_train, y_train
            )

            if x_train_processed.size < 2:
                print(
                    f"Warning: After preprocessing, not enough unique data points for '{relationship_name}'. Skipping."
                )
                continue

            x_interpolation_range = self._create_interpolation_range(
                x_train_processed.min(), x_train_processed.max()
            )

            for (
                method_name,
                method_class_or_kind,
            ) in self.INTERPOLATION_METHODS_1D.items():
                try:
                    interp_func = None
                    if method_name == "RBF":
                        # RBFInterpolator requires (N, D) input. For 1D, D=1.
                        interp_func = method_class_or_kind(
                            np.atleast_2d(x_train_processed).T, y_train_processed
                        )
                    elif isinstance(method_class_or_kind, str):  # For interp1d 'kind'
                        interp_func = spi.interp1d(
                            x_train_processed,
                            y_train_processed,
                            kind=method_class_or_kind,
                            fill_value="extrapolate",
                            assume_sorted=True,  # Data is already sorted and unique
                        )
                    else:  # For classes like PchipInterpolator, CubicSpline
                        interp_func = method_class_or_kind(
                            x_train_processed, y_train_processed
                        )

                    # Predict interpolated values. RBF needs 2D input for prediction as well.
                    predict_x = (
                        np.atleast_2d(x_interpolation_range).T
                        if method_name == "RBF"
                        else x_interpolation_range
                    )
                    interpolated_y_values = interp_func(predict_x)

                    data.interpolations_1d[relationship_name][method_name] = (
                        x_interpolation_range,
                        interpolated_y_values,
                    )

                except Exception as e:
                    print(
                        f"Error computing 1D interpolation for {relationship_name} with '{method_name}': {e}. Skipping."
                    )
                    continue

    def _create_interpolation_range(self, min_val: float, max_val: float) -> np.ndarray:
        """Helper to create a linspace for interpolation."""
        if min_val == max_val:
            return np.array([min_val])
        return np.linspace(min_val, max_val, self._NUM_INTERPOLATION_POINTS_1D)

    def _compute_all_2d_interpolations(self, data: ParetoData):
        """Fit and predict 2D scipy.interpolate mappers for various relationships."""
        # Using the direct attributes from ParetoData
        relationship_data_map = {
            "f1f2_vs_x1": {
                "X_train": np.column_stack((data.norm_f1, data.norm_f2)),
                "y_train": data.norm_x1,
            },
            "f1f2_vs_x2": {
                "X_train": np.column_stack((data.norm_f1, data.norm_f2)),
                "y_train": data.norm_x2,
            },
        }

        for relationship_name, data_sources in relationship_data_map.items():
            X_train = data_sources["X_train"]
            y_train = data_sources["y_train"]

            if X_train.shape[0] < 2 or y_train.size < 2:
                print(
                    f"Warning: Not enough data points to interpolate for '{relationship_name}'. Skipping."
                )
                continue

            # Create the 2D grid for prediction
            f1_min, f1_max = X_train[:, 0].min(), X_train[:, 0].max()
            f2_min, f2_max = X_train[:, 1].min(), X_train[:, 1].max()

            grid_f1 = (
                np.linspace(f1_min, f1_max, self._NUM_INTERPOLATION_POINTS_2D_GRID)
                if f1_min != f1_max
                else np.array([f1_min])
            )
            grid_f2 = (
                np.linspace(f2_min, f2_max, self._NUM_INTERPOLATION_POINTS_2D_GRID)
                if f2_min != f2_max
                else np.array([f2_min])
            )

            mesh_f1, mesh_f2 = np.meshgrid(grid_f1, grid_f2)
            points_for_prediction_2d = np.column_stack(
                (mesh_f1.ravel(), mesh_f2.ravel())
            )

            for method_name, method_class in self.INTERPOLATION_METHODS_ND.items():
                try:
                    interp_func = method_class(X_train, y_train)

                    interpolated_z_values_flat = interp_func(points_for_prediction_2d)
                    interpolated_z_values = interpolated_z_values_flat.reshape(
                        mesh_f1.shape
                    )

                    data.interpolations_2d[relationship_name][method_name] = (
                        mesh_f1,
                        mesh_f2,
                        interpolated_z_values,
                    )

                except Exception as e:
                    print(
                        f"Error computing 2D interpolation for {relationship_name} with '{method_name}': {e}. Skipping."
                    )
                    continue


class PlotlyParetoDataVisualizer(BaseDataVisualizer):
    """
    Dashboard for visualizing Pareto set and front with precomputed interpolations.
    Displays multi-objective optimization results, including decision space,
    objective space, normalized spaces, and various interpolation visualizations.
    """

    _INTERPOLATION_COLORS: dict[str, str] = {
        "Pchip": "#07FF03",
        "Cubic Spline": "#CD05F9",
        "Linear": "#43A047",
        "Quadratic": "#7B1FA2",
        "RBF": "#F30B0B",
        "Nearest Neighbor": "#F57C00",
        "Linear ND": "#FFEA07",
    }

    _SUBPLOT_CONFIG: dict[tuple[int, int], dict[str, Any]] = {
        (1, 1): {
            "type": "scatter",
            "title": "Decision Space ($x_1$ vs $x_2$)",
            "x_label": "$x_1$",
            "y_label": "$x_2$",
            "data_key": "pareto_set",
            "data_mapping": {"x": 0, "y": 1},
            "name": "Pareto Set",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 7,
            "showlegend": True,
        },
        (1, 2): {
            "type": "scatter",
            "title": "Objective Space ($f_1$ vs $f_2$)",
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
        (2, 1): {
            "type": "scatter",
            "title": "Normalized Decision Space",
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
            "x_data_attr": "norm_x1",
            "y_data_attr": "norm_x2",
            "name": "Norm Pareto Set",
            "color": "#3498db",
            "symbol": "diamond",
            "marker_size": 6,
            "showlegend": True,
        },
        (2, 2): {
            "type": "scatter",
            "title": "Normalized Objective Space",
            "x_label": "Norm $f_1$",
            "y_label": "Norm $f_2$",
            "x_data_attr": "norm_f1",
            "y_data_attr": "norm_f2",
            "name": "Norm Pareto Front",
            "color": "#3498db",
            "symbol": "diamond",
            "marker_size": 6,
            "showlegend": True,
        },
        (3, 1): {
            "type": "scatter",
            "title": "$x_1$ vs $x_2$ (Interpolations)",
            "x_label": "Normalized $x_1$",
            "y_label": "Normalized $x_2$",
            "x_data_attr": "norm_x1",
            "y_data_attr": "norm_x2",
            "interpolation_source_key": "interpolations_1d",
            "interpolation_relationship_key": "x1_vs_x2",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
        },
        (3, 2): {
            "type": "scatter",
            "title": "$f_1$ vs $f_2$ (Interpolations)",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $f_2$",
            "x_data_attr": "norm_f1",
            "y_data_attr": "norm_f2",
            "interpolation_source_key": "interpolations_1d",
            "interpolation_relationship_key": "f1_vs_f2",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
        },
        (4, 1): {
            "type": "scatter",
            "title": "$f_1$ vs $x_1$ (Interpolations)",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $x_1$",
            "x_data_attr": "norm_f1",
            "y_data_attr": "norm_x1",
            "interpolation_source_key": "interpolations_1d",
            "interpolation_relationship_key": "f1_vs_x1",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
        },
        (4, 2): {
            "type": "scatter",
            "title": "$f_1$ vs $x_2$ (Interpolations)",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $x_2$",
            "x_data_attr": "norm_f1",
            "y_data_attr": "norm_x2",
            "interpolation_source_key": "interpolations_1d",
            "interpolation_relationship_key": "f1_vs_x2",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
        },
        (5, 1): {
            "type": "scatter",
            "title": "$f_2$ vs $x_1$ (Interpolations)",
            "x_label": "Normalized $f_2$",
            "y_label": "Normalized $x_1$",
            "x_data_attr": "norm_f2",
            "y_data_attr": "norm_x1",
            "interpolation_source_key": "interpolations_1d",
            "interpolation_relationship_key": "f2_vs_x1",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
        },
        (5, 2): {
            "type": "scatter",
            "title": "$f_2$ vs $x_2$ (Interpolations)",
            "x_label": "Normalized $f_2$",
            "y_label": "Normalized $x_2$",
            "x_data_attr": "norm_f2",
            "y_data_attr": "norm_x2",
            "interpolation_source_key": "interpolations_1d",
            "interpolation_relationship_key": "f2_vs_x2",
            "name": "Data Points",
            "color": "#3498db",
            "symbol": "circle",
            "marker_size": 6,
            "showlegend": False,
        },
        (6, 1): {
            "type": "scatter3d",
            "title": "3D: $f_1$, $f_2$, $x_1$",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
            "z_label": "$x_1$",
            "x_data_attr": "norm_f1",
            "y_data_attr": "norm_f2",
            "z_data_attr": "norm_x1",
            "surface_source_key": "interpolations_2d",
            "surface_relationship_key": "f1f2_vs_x1",
            "name": "Original Points",
            "color": "#1f77b4",
            "marker_size": 5,
        },
        (6, 2): {
            "type": "scatter3d",
            "title": "3D: $f_1$, $f_2$, $x_2$",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
            "z_label": "$x_2$",
            "x_data_attr": "norm_f1",
            "y_data_attr": "norm_f2",
            "z_data_attr": "norm_x2",
            "surface_source_key": "interpolations_2d",
            "surface_relationship_key": "f1f2_vs_x2",
            "name": "Original Points",
            "color": "#1f77b4",
            "marker_size": 5,
        },
    }

    _FIGURE_LAYOUT_CONFIG: dict[str, Any] = {
        "title_text": "Enhanced Pareto Optimization Dashboard",
        "title_x": 0.5,
        "title_font_size": 24,
        "height": 2700,
        "width": 1600,
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
        "hovermode": "closest",
    }

    # Description positions are updated for the new 6-row layout.
    _DESCRIPTION_POSITIONS = {
        "row": {1: 0.97, 2: 0.82, 3: 0.67, 4: 0.52, 5: 0.37, 6: 0.15},
        "col": {1: 0.25, 2: 0.75},
    }

    def __init__(self, save_path: Path | None = None):
        """
        Initializes the PlotlyParetoDataVisualizer.

        Args:
            save_path (Path | None): Optional path to save the generated plots.
        """
        super().__init__(save_path)
        self._added_legend_items = set()
        self._data_service = ParetoDataService()
        self._save_path = False

    def plot(self, data: Any):
        """
        Generates and displays a comprehensive Pareto optimization dashboard.

        Args:
            dto (ParetoVisualizationDTO): Data Transfer Object containing all
                                          pre-processed data for visualization.
        """

        data = self._data_service.prepare_data(data)

        fig = self._create_figure_layout()
        self._add_all_subplots_from_config(fig, data)

        fig.show()

        if self._save_path:
            self._save_plot(fig)

    def _save_plot(self, fig: go.Figure) -> None:
        """Saves the generated Plotly figure to the specified path."""
        try:
            fig.write_html(str(self._save_path))
            print(f"Plot saved successfully to {self._save_path}")
        except Exception as e:
            print(f"Error saving plot to {self._save_path}: {e}")

    def _create_figure_layout(self) -> go.Figure:
        """
        Creates and configures the main figure layout for the dashboard.
        """
        rows = max(r for r, c in self._SUBPLOT_CONFIG.keys())
        cols = 2

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
            horizontal_spacing=0.05,
            vertical_spacing=0.05,
            column_widths=[0.5, 0.5],
            row_heights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
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
            hovermode="closest",
        )
        return fig

    def _add_all_subplots_from_config(self, fig: go.Figure, data: ParetoData) -> None:
        """
        Iterates through _SUBPLOT_CONFIG and adds traces dynamically based on the ParetoData object.
        """
        for (row, col), config in self._SUBPLOT_CONFIG.items():
            plot_type = config.get("type")

            if plot_type == "scatter":
                if "data_key" in config:
                    data_source = getattr(data, config["data_key"], None)
                    x_data = self._get_data_from_indexed_source(
                        data_source, config.get("data_mapping", {}).get("x")
                    )
                    y_data = self._get_data_from_indexed_source(
                        data_source, config.get("data_mapping", {}).get("y")
                    )
                else:
                    x_data = getattr(data, config.get("x_data_attr"), None)
                    y_data = getattr(data, config.get("y_data_attr"), None)

                if (
                    x_data is None
                    or y_data is None
                    or x_data.size == 0
                    or y_data.size == 0
                ):
                    print(
                        f"Warning: Insufficient data for {config.get('title', 'scatter plot')} at ({row}, {col}). Skipping."
                    )
                    continue

                self._add_scatter_plot(fig, row, col, x_data, y_data, config)

                if (
                    "interpolation_source_key" in config
                    and "interpolation_relationship_key" in config
                ):
                    interpolations_dict = getattr(
                        data, config["interpolation_source_key"], {}
                    )
                    method_interpolations = interpolations_dict.get(
                        config["interpolation_relationship_key"], {}
                    )

                    if method_interpolations:
                        self._add_1d_interpolation_traces(
                            fig, row, col, method_interpolations, config
                        )

            elif plot_type == "parcoords":
                parcoords_data = data.parallel_coordinates_data
                if parcoords_data is None or parcoords_data.size == 0:
                    print(
                        f"Warning: Missing or empty data for parallel coordinates plot at ({row}, {col}). Skipping."
                    )
                    continue
                # self._add_parcoords_plot(fig, row, col, parcoords_data, config)

            elif plot_type == "scatter3d":
                x_data = getattr(data, config.get("x_data_attr"), None)
                y_data = getattr(data, config.get("y_data_attr"), None)
                z_data = getattr(data, config.get("z_data_attr"), None)

                if (
                    x_data is None
                    or y_data is None
                    or z_data is None
                    or x_data.size == 0
                    or y_data.size == 0
                    or z_data.size == 0
                ):
                    print(
                        f"Warning: Insufficient data for {config.get('title', '3D scatter plot')} at ({row}, {col}). Skipping."
                    )
                    continue

                self._add_scatter3d_plot(fig, row, col, x_data, y_data, z_data, config)

                if (
                    "surface_source_key" in config
                    and "surface_relationship_key" in config
                ):
                    surface_interpolations_dict = getattr(
                        data, config["surface_source_key"], {}
                    )
                    method_surface_interpolations = surface_interpolations_dict.get(
                        config["surface_relationship_key"], {}
                    )

                    if method_surface_interpolations:
                        self._add_2d_surface_traces(
                            fig, row, col, method_surface_interpolations, config
                        )
            else:
                print(
                    f"Warning: Unsupported plot type '{plot_type}' for subplot ({row}, {col})."
                )

    def _get_data_from_indexed_source(
        self, data_source: np.ndarray | tuple, index: int | None
    ) -> np.ndarray | None:
        """
        Helper to retrieve data from a numpy array or tuple using an integer index.
        Used for original pareto_set/front which are 2D arrays.
        """
        if data_source is None or index is None:
            return None

        if isinstance(data_source, np.ndarray):
            if data_source.ndim > 1:
                return data_source[:, index]
            else:
                return data_source if index == 0 else None
        elif isinstance(data_source, (list, tuple)):
            try:
                return np.asarray(data_source[index])
            except (IndexError, TypeError):
                print(
                    f"Error: Invalid index {index} for data source type {type(data_source)}."
                )
                return None
        return None

    def _add_scatter_plot(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        config: dict[str, Any],
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
                hovertemplate=f"{config['x_label']}: %{{x:.4f}}<br>{config['y_label']}: %{{y:.4f}}<extra>{config['name']}</extra>",
            ),
            row=row,
            col=col,
        )
        self._set_axis_limits(fig, row, col, x, y)
        fig.update_xaxes(title_text=config["x_label"], row=row, col=col)
        fig.update_yaxes(title_text=config["y_label"], row=row, col=col)

    def _add_1d_interpolation_traces(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        interpolation_methods_dict: dict[str, tuple[np.ndarray, np.ndarray]],
        config: dict[str, Any],
    ) -> None:
        """
        Adds 1D interpolation lines for a scatter plot, iterating through all methods in the dict.
        """
        for method_name, (x_grid, y_grid) in interpolation_methods_dict.items():
            show_legend_item = method_name not in self._added_legend_items
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    line=dict(
                        color=self._INTERPOLATION_COLORS.get(method_name, "#000000"),
                        width=2.5,
                        dash="solid",
                    ),
                    name=method_name,
                    legendgroup=method_name,
                    showlegend=show_legend_item,
                    hovertemplate=f"Method: {method_name}<br>{config['x_label']}: %{{x:.4f}}<br>{config['y_label']}: %{{y:.4f}}<extra></extra>",
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
        config: dict[str, Any],
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
                hovertemplate=f"{config['x_label']}: %{{x:.4f}}<br>{config['y_label']}: %{{y:.4f}}<br>{config['z_label']}: %{{z:.4f}}<extra>{config['name']}</extra>",
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
            aspectmode="data",
            row=row,
            col=col,
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
        )

    def _add_2d_surface_traces(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        surface_methods_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        config: dict[str, Any],
    ) -> None:
        """
        Adds 3D surface interpolations, iterating through all methods in the dict.
        """
        for method_name, (X_grid, Y_grid, Z_grid) in surface_methods_dict.items():
            show_in_legend = method_name not in self._added_legend_items

            fig.add_trace(
                go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale=[
                        [0, self._INTERPOLATION_COLORS.get(method_name, "#000000")],
                        [1, self._INTERPOLATION_COLORS.get(method_name, "#000000")],
                    ],
                    opacity=0.7,
                    name=method_name,
                    showlegend=show_in_legend,
                    showscale=False,
                    legendgroup=method_name,
                    hovertemplate=f"Method: {method_name}<br>{config['x_label']}: %{{x:.4f}}<br>{config['y_label']}: %{{y:.4f}}<br>{config['z_label']}: %{{z:.4f}}<extra></extra>",
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
