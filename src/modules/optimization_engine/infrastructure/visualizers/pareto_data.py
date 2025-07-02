from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import scipy.interpolate as spi
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

from ...domain.analyzing.interfaces.base_visualizer import BaseDataVisualizer


@dataclass
class ParetoDataset:
    """
    Container for all Pareto optimization data in various representations:
    - Core Pareto set and front (original values)
    - Normalized values
    - Computed interpolations
    """

    # --- Core Pareto Data ---
    pareto_set: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={
            "description": "Original decision variables (X) for Pareto optimal solutions. Expected shape (n_samples, n_decision_vars)."
        },
    )
    pareto_front: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={
            "description": "Original objective function values (F) for Pareto optimal solutions. Expected shape (n_samples, n_objective_vars)."
        },
    )

    # --- Normalized Values (0-1 range) ---
    normalized: dict[str, np.ndarray | None] = field(
        default_factory=lambda: {"f1": None, "f2": None, "x1": None, "x2": None},
        metadata={
            "description": "Dictionary of normalized (0-1 range) decision and objective variables."
        },
    )

    @property
    def norm_f1(self) -> np.ndarray:
        """Returns the normalized values of objective function 1."""
        f1 = self.normalized.get("f1")
        if f1 is None or f1.size == 0:
            raise AttributeError("Normalized 'f1' data is not set in ParetoDataset.")
        return f1

    @property
    def norm_f2(self) -> np.ndarray:
        """Returns the normalized values of objective function 2."""
        f2 = self.normalized.get("f2")
        if f2 is None or f2.size == 0:
            raise AttributeError("Normalized 'f2' data is not set in ParetoDataset.")
        return f2

    @property
    def norm_x1(self) -> np.ndarray:
        """Returns the normalized values of decision variable 1."""
        x1 = self.normalized.get("x1")
        if x1 is None or x1.size == 0:
            raise AttributeError("Normalized 'x1' data is not set in ParetoDataset.")
        return x1

    @property
    def norm_x2(self) -> np.ndarray:
        """Returns the normalized values of decision variable 2."""
        x2 = self.normalized.get("x2")
        if x2 is None or x2.size == 0:
            raise AttributeError("Normalized 'x2' data is not set in ParetoDataset.")
        return x2

    # --- 1D Interpolations ---
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

    def get_1d_interpolated_data(
        self, relationship: str, method: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the specific 1D interpolated data for a given relationship and method.

        Args:
            relationship (str): The name of the relationship (e.g., "f1_vs_f2").
            method (str): The name of the interpolation method (e.g., "Pchip", "Cubic Spline").

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the x and y interpolated data arrays.

        Raises:
            KeyError: If the relationship or method is not found.
        """
        if relationship not in self.interpolations_1d:
            raise KeyError(
                f"1D interpolated data relationship '{relationship}' not found."
            )
        if method not in self.interpolations_1d[relationship]:
            raise KeyError(
                f"Interpolated data for method '{method}' not found for relationship '{relationship}'."
            )
        return self.interpolations_1d[relationship][method]

    # --- 2D Multivariate Interpolations ---
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

    def get_2d_interpolated_data(
        self, relationship: str, method: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves the specific 2D interpolated data for a given relationship and method.

        Args:
            relationship (str): The name of the relationship (e.g., "f1f2_vs_x1").
            method (str): The name of the mapping method (e.g., "Linear ND", "Nearest Neighbor").

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the x_grid_1, x_grid_2, and z_values arrays.

        Raises:
            KeyError: If the relationship or method is not found.
        """
        if relationship not in self.interpolations_2d:
            raise KeyError(
                f"2D interpolated data relationship '{relationship}' not found."
            )
        if method not in self.interpolations_2d[relationship]:
            raise KeyError(
                f"Interpolated data for method '{method}' not found for relationship '{relationship}'."
            )
        return self.interpolations_2d[relationship][method]


class ParetoDataService:
    """
    Service for preparing Pareto optimization data in various representations.
    This version creates and fits interpolators from scipy.interpolate directly.
    """

    # --- Updated Interpolation Methods to use scipy.interpolate classes/kinds directly ---
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

    def __init__(self):
        # The repository dependency is removed as we now create interpolators directly.
        pass

    def prepare_dataset(self, pareto_data: Any) -> ParetoDataset:
        if not hasattr(pareto_data, "pareto_set") or not hasattr(
            pareto_data, "pareto_front"
        ):
            raise ValueError(
                "Archiver did not return valid Pareto data (missing pareto_set or pareto_front)."
            )

        if pareto_data.pareto_set.size == 0 or pareto_data.pareto_front.size == 0:
            raise ValueError(
                "Loaded Pareto data (pareto_set or pareto_front) is empty."
            )

        dataset = ParetoDataset()
        dataset.pareto_set = pareto_data.pareto_set
        dataset.pareto_front = pareto_data.pareto_front

        if dataset.pareto_front.shape[1] < 2:
            raise ValueError("pareto_front must have at least 2 columns for f1 and f2.")
        if dataset.pareto_set.shape[1] < 2:
            raise ValueError("pareto_set must have at least 2 columns for x1 and x2.")

        dataset.normalized["f1"] = self._normalize_array(dataset.pareto_front[:, 0])
        dataset.normalized["f2"] = self._normalize_array(dataset.pareto_front[:, 1])
        dataset.normalized["x1"] = self._normalize_array(dataset.pareto_set[:, 0])
        dataset.normalized["x2"] = self._normalize_array(dataset.pareto_set[:, 1])

        self._compute_all_1d_interpolations(dataset)
        self._compute_all_2d_interpolations(dataset)

        return dataset

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
        # Sort data by x values
        sort_indices = np.argsort(x)
        x_sorted = x[sort_indices]
        y_sorted = y[sort_indices]

        # Find indices of unique x values to remove duplicates
        _, unique_indices = np.unique(x_sorted, return_index=True)

        # Return the data with unique x values
        return x_sorted[unique_indices], y_sorted[unique_indices]

    def _compute_all_1d_interpolations(self, dataset: ParetoDataset):
        """Fit and predict 1D scipy.interpolate mappers for various relationships."""
        norm = dataset.normalized

        # --- Define the data mapping for each relationship ---
        relationship_data_map = {
            "f1_vs_f2": {"x_train": norm["f1"], "y_train": norm["f2"]},
            "f1_vs_x1": {"x_train": norm["f1"], "y_train": norm["x1"]},
            "f1_vs_x2": {"x_train": norm["f1"], "y_train": norm["x2"]},
            "x1_vs_x2": {"x_train": norm["x1"], "y_train": norm["x2"]},
            "f2_vs_x1": {"x_train": norm["f2"], "y_train": norm["x1"]},
            "f2_vs_x2": {"x_train": norm["f2"], "y_train": norm["x2"]},
        }

        for relationship_name, data_sources in relationship_data_map.items():
            x_train = data_sources["x_train"]
            y_train = data_sources["y_train"]

            # Ensure data is valid for interpolation
            if (
                x_train is None
                or y_train is None
                or x_train.size < 2
                or y_train.size < 2
            ):
                print(
                    f"Warning: Not enough data points to interpolate for '{relationship_name}'. Skipping."
                )
                continue

            # --- Preprocess data to be sorted and have unique x values ---
            x_train_processed, y_train_processed = self._preprocess_1d_data(
                x_train, y_train
            )

            if x_train_processed.size < 2:
                print(
                    f"Warning: After preprocessing, not enough unique data points for '{relationship_name}'. Skipping."
                )
                continue

            # Create the interpolation range from the processed training data's min/max
            x_interpolation_range = self._create_interpolation_range(
                x_train_processed.min(), x_train_processed.max()
            )

            for (
                method_name,
                method_class_or_kind,
            ) in self.INTERPOLATION_METHODS_1D.items():
                try:
                    interpolator_input_x = x_train_processed
                    interpolator_predict_x = x_interpolation_range

                    # --- Special handling for RBFInterpolator's 2D input requirement ---
                    if method_name == "RBF":
                        # RBFInterpolator requires the input data points `x` to be 2D, shape (n_points, n_dims).
                        # We use np.atleast_2d and transpose to guarantee a (N, 1) shape for 1D data.
                        interpolator_input_x = np.atleast_2d(x_train_processed).T
                        interpolator_predict_x = np.atleast_2d(x_interpolation_range).T

                        interp_func = method_class_or_kind(
                            interpolator_input_x, y_train_processed
                        )

                    # --- Standard handling for other 1D interpolators ---
                    elif isinstance(method_class_or_kind, str):  # For interp1d 'kind'
                        # `interp1d` requires a sorted x, but not necessarily unique for 'linear'
                        # 'quadratic' and 'cubic' do need unique, so preprocessing helps all.
                        interp_func = spi.interp1d(
                            interpolator_input_x,
                            y_train_processed,
                            kind=method_class_or_kind,
                            fill_value="extrapolate",
                        )
                    else:  # For classes like PchipInterpolator, CubicSpline
                        interp_func = method_class_or_kind(
                            interpolator_input_x, y_train_processed
                        )

                    # Predict interpolated values
                    interpolated_y_values = interp_func(interpolator_predict_x)

                    # Store the result in the dataset
                    dataset.interpolations_1d[relationship_name][method_name] = (
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

    def _compute_all_2d_interpolations(self, dataset: ParetoDataset):
        """Fit and predict 2D scipy.interpolate mappers for various relationships."""
        norm = dataset.normalized

        # --- Define the data mapping for each relationship ---
        relationship_data_map = {
            "f1f2_vs_x1": {
                "X_train": np.column_stack((norm["f1"], norm["f2"])),
                "y_train": norm["x1"],
            },
            "f1f2_vs_x2": {
                "X_train": np.column_stack((norm["f1"], norm["f2"])),
                "y_train": norm["x2"],
            },
        }

        for relationship_name, data_sources in relationship_data_map.items():
            X_train = data_sources["X_train"]
            y_train = data_sources["y_train"]

            # Ensure data is valid for interpolation
            if (
                X_train is None
                or y_train is None
                or X_train.shape[0] < 2
                or y_train.size < 2
            ):
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
                    # Create and fit the interpolator instance
                    interp_func = method_class(X_train, y_train)

                    # Predict interpolated Z values
                    interpolated_z_values_flat = interp_func(points_for_prediction_2d)
                    interpolated_z_values = interpolated_z_values_flat.reshape(
                        mesh_f1.shape
                    )

                    # Store the result in the dataset
                    dataset.interpolations_2d[relationship_name][method_name] = (
                        mesh_f1,
                        mesh_f2,
                        interpolated_z_values,
                    )

                except Exception as e:
                    print(
                        f"Error computing 2D interpolation for {relationship_name} with '{method_name}': {e}. Skipping."
                    )
                    continue


@dataclass
class ParetoVisualizationDTO:
    """
    Data Transfer Object for Pareto visualization data.
    Provides a strict interface for the visualizer with validated data structure.

    This version is designed to align with the ParetoDataset's nested
    structure for interpolations, allowing for multiple interpolation methods.
    """

    pareto_set: np.ndarray  # Original X values (decision space)
    pareto_front: np.ndarray  # Original F values (objective space)

    # Individual normalized components, directly accessible at the top level
    norm_x1: np.ndarray
    norm_x2: np.ndarray
    norm_f1: np.ndarray
    norm_f2: np.ndarray

    # Data for parallel coordinates (combined normalized data)
    parallel_coordinates_data: np.ndarray

    # Dictionaries containing interpolation grids, now mirroring ParetoDataset's structure:
    # Outer dict: relationship_name -> Inner dict: method_name -> (x_grid, y_grid)
    interpolations_1d: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]

    # Outer dict: relationship_name -> Inner dict: method_name -> (X_grid, Y_grid, Z_grid)
    interpolations_2d: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]


class ParetoVisualizationMapper:
    """
    Maps ParetoDataset to a structured DTO suitable for visualization.
    Performs data validation and transformation.
    """

    def map_to_dto(
        self, dataset: ParetoDataset
    ) -> ParetoVisualizationDTO:  # Changed type hint to ParetoDataset
        """
        Transform dataset into visualization DTO
        """
        # Orchestrate validation using dedicated methods
        # The validation methods will now rely on the properties for their checks where applicable
        self._validate_core_data(dataset)
        self._validate_normalized_data(
            dataset
        )  # This method will be updated to use properties internally
        self._validate_1d_interpolations(dataset)
        self._validate_2d_interpolations(dataset)

        # Extract normalized components for direct access in DTO
        # Now using the properties for cleaner and safer access
        norm_x1 = dataset.norm_x1
        norm_x2 = dataset.norm_x2
        norm_f1 = dataset.norm_f1
        norm_f2 = dataset.norm_f2

        return ParetoVisualizationDTO(
            pareto_set=dataset.pareto_set,
            pareto_front=dataset.pareto_front,
            norm_x1=norm_x1,
            norm_x2=norm_x2,
            norm_f1=norm_f1,
            norm_f2=norm_f2,
            # Parallel coordinates data uses the normalized values
            parallel_coordinates_data=np.hstack(
                [
                    norm_x1.reshape(-1, 1),
                    norm_x2.reshape(-1, 1),
                    norm_f1.reshape(-1, 1),
                    norm_f2.reshape(-1, 1),
                ]
            ),
            interpolations_1d=dataset.interpolations_1d,
            interpolations_2d=dataset.interpolations_2d,
        )

    def _validate_core_data(self, dataset: ParetoDataset):
        """Validates the core Pareto set and front arrays, including their dimensions."""
        # No change here, pareto_set and pareto_front are direct attributes
        if (
            not isinstance(dataset.pareto_set, np.ndarray)
            or dataset.pareto_set.size == 0
        ):
            raise ValueError(
                "Invalid or missing 'pareto_set' in dataset. Must be a non-empty numpy array."
            )
        if dataset.pareto_set.ndim != 2 or dataset.pareto_set.shape[1] < 2:
            raise ValueError(
                f"'pareto_set' must be a 2D array with at least 2 columns (for x1, x2), but has shape {dataset.pareto_set.shape}."
            )

        if (
            not isinstance(dataset.pareto_front, np.ndarray)
            or dataset.pareto_front.size == 0
        ):
            raise ValueError(
                "Invalid or missing 'pareto_front' in dataset. Must be a non-empty numpy array."
            )
        if dataset.pareto_front.ndim != 2 or dataset.pareto_front.shape[1] < 2:
            raise ValueError(
                f"'pareto_front' must be a 2D array with at least 2 columns (for f1, f2), but has shape {dataset.pareto_front.shape}."
            )

        if dataset.pareto_set.shape[0] != dataset.pareto_front.shape[0]:
            raise ValueError(
                f"Number of samples in 'pareto_set' ({dataset.pareto_set.shape[0]}) does not match 'pareto_front' ({dataset.pareto_front.shape[0]})."
            )

    def _validate_normalized_data(self, dataset: ParetoDataset):
        """
        Validates the normalized data using ParetoDataset's properties.
        This implicitly checks the existence and type through the property access.
        """
        # Attempt to access properties to trigger AttributeError if data is None/missing
        try:
            norm_f1 = dataset.norm_f1
            norm_f2 = dataset.norm_f2
            norm_x1 = dataset.norm_x1
            norm_x2 = dataset.norm_x2
        except AttributeError as e:
            raise ValueError(
                f"Missing or unset normalized data in ParetoDataset: {e}"
            ) from e

        # Further dimension and size checks on the retrieved arrays
        if not isinstance(norm_f1, np.ndarray) or norm_f1.ndim != 1:
            raise TypeError("Normalized 'f1' should be a 1D numpy array.")
        if not isinstance(norm_f2, np.ndarray) or norm_f2.ndim != 1:
            raise TypeError("Normalized 'f2' should be a 1D numpy array.")
        if not isinstance(norm_x1, np.ndarray) or norm_x1.ndim != 1:
            raise TypeError("Normalized 'x1' should be a 1D numpy array.")
        if not isinstance(norm_x2, np.ndarray) or norm_x2.ndim != 1:
            raise TypeError("Normalized 'x2' should be a 1D numpy array.")

        # Check sample count consistency after confirming they are arrays
        expected_samples = dataset.pareto_set.shape[0]
        if norm_f1.shape[0] != expected_samples:
            raise ValueError(
                f"Normalized 'f1' sample count ({norm_f1.shape[0]}) does not match 'pareto_set' ({expected_samples})."
            )
        if norm_f2.shape[0] != expected_samples:
            raise ValueError(
                f"Normalized 'f2' sample count ({norm_f2.shape[0]}) does not match 'pareto_set' ({expected_samples})."
            )
        if norm_x1.shape[0] != expected_samples:
            raise ValueError(
                f"Normalized 'x1' sample count ({norm_x1.shape[0]}) does not match 'pareto_set' ({expected_samples})."
            )
        if norm_x2.shape[0] != expected_samples:
            raise ValueError(
                f"Normalized 'x2' sample count ({norm_x2.shape[0]}) does not match 'pareto_set' ({expected_samples})."
            )

    def _validate_1d_interpolations(self, dataset: ParetoDataset):
        """
        Validates the 1D interpolations dictionary.
        This method will continue to access the dictionary directly as it's
        dealing with the structure of the `interpolations_1d` field itself,
        not individual pre-extracted components like `norm_x1`.
        However, we can leverage `get_1d_interpolation` if we wanted to
        validate specific relationships/methods, but here we're validating
        the *container* structure.
        """
        required_1d_rel_keys = [
            "f1_vs_f2",
            "f1_vs_x1",
            "f1_vs_x2",
            "x1_vs_x2",
            "f2_vs_x1",
            "f2_vs_x2",
        ]
        if not isinstance(dataset.interpolations_1d, dict):
            raise TypeError("'dataset.interpolations_1d' is not a dictionary.")

        for rel_key in required_1d_rel_keys:
            if rel_key not in dataset.interpolations_1d:
                raise ValueError(
                    f"Missing 1D interpolation relationship key: '{rel_key}' in 'dataset.interpolations_1d'."
                )

            if not isinstance(dataset.interpolations_1d[rel_key], dict):
                raise TypeError(
                    f"1D interpolation relationship '{rel_key}' is not a dictionary of methods."
                )

            for method_name, grid_tuple in dataset.interpolations_1d[rel_key].items():
                if not (
                    isinstance(grid_tuple, tuple)
                    and len(grid_tuple) == 2
                    and all(isinstance(arr, np.ndarray) for arr in grid_tuple)
                    and all(arr.ndim == 1 for arr in grid_tuple)
                ):
                    raise TypeError(
                        f"1D interpolation '{rel_key}' method '{method_name}' must be a tuple of two 1D numpy arrays."
                    )

    def _validate_2d_interpolations(self, dataset: ParetoDataset):
        """
        Validates the 2D interpolations dictionary.
        Similar to _validate_1d_interpolations, this checks the container structure.
        """
        required_2d_rel_keys = ["f1f2_vs_x1", "f1f2_vs_x2"]
        if not isinstance(dataset.interpolations_2d, dict):
            raise TypeError("'dataset.interpolations_2d' is not a dictionary.")

        for rel_key in required_2d_rel_keys:
            if rel_key not in dataset.interpolations_2d:
                raise ValueError(
                    f"Missing 2D interpolation relationship key: '{rel_key}' in 'dataset.interpolations_2d'."
                )

            if not isinstance(dataset.interpolations_2d[rel_key], dict):
                raise TypeError(
                    f"2D interpolation relationship '{rel_key}' is not a dictionary of methods."
                )

            for method_name, grid_tuple in dataset.interpolations_2d[rel_key].items():
                if not (
                    isinstance(grid_tuple, tuple)
                    and len(grid_tuple) == 3
                    and all(isinstance(arr, np.ndarray) for arr in grid_tuple)
                    and all(arr.ndim >= 2 for arr in grid_tuple)
                ):
                    raise TypeError(
                        f"2D interpolation '{rel_key}' method '{method_name}' must be a tuple of three (>=2D) numpy arrays (X, Y, Z grids)."
                    )


class PlotlyParetoDataVisualizer(BaseDataVisualizer):
    """
    Dashboard for visualizing Pareto set and front with precomputed interpolations.

    This class orchestrates the creation of a comprehensive Plotly dashboard
    to display multi-objective optimization results, including decision space,
    objective space, parallel coordinates, normalized spaces, and
    various interpolation visualizations.
    """

    _INTERPOLATION_COLORS = {
        "Pchip": "#07FF03",
        "Cubic Spline": "#CD05F9",
        "Linear": "#43A047",
        "Quadratic": "#7B1FA2",
        "RBF": "#F30B0B",
        "Nearest Neighbor": "#F57C00",
        "Linear ND": "#FFEA07",
    }

    # Updated for a 2-column layout with the parallel coordinates plot removed.
    # The plots have been re-packed to fill the grid.
    _SUBPLOT_CONFIG = {
        (1, 1): {
            "type": "scatter",
            "title": "Decision Space ($x_1$ vs $x_2$)",
            "description": "Original decision variables showing trade-offs",
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
        # Removed the parallel coordinates plot at (2, 1)
        (2, 1): {
            "type": "scatter",
            "title": "Normalized Decision Space",
            "description": "Decision variables scaled to [0,1]",
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
            "x_data_key": "norm_x1",
            "y_data_key": "norm_x2",
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
            "x_data_key": "norm_f1",
            "y_data_key": "norm_f2",
            "name": "Norm Pareto Front",
            "color": "#3498db",
            "symbol": "diamond",
            "marker_size": 6,
            "showlegend": True,
        },
        (3, 1): {
            "type": "scatter",
            "title": "$x_1$ vs $x_2$ (Interpolations)",
            "description": "Relationship between decision variables",
            "x_label": "Normalized $x_1$",
            "y_label": "Normalized $x_2$",
            "x_data_key": "norm_x1",
            "y_data_key": "norm_x2",
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
            "description": "Relationship between $f_1$ and $f_2$",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $f_2$",
            "x_data_key": "norm_f1",
            "y_data_key": "norm_f2",
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
            "description": "Relationship between $f_1$ and $x_1$",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $x_1$",
            "x_data_key": "norm_f1",
            "y_data_key": "norm_x1",
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
            "description": "Relationship between $f_1$ and $x_2$",
            "x_label": "Normalized $f_1$",
            "y_label": "Normalized $x_2$",
            "x_data_key": "norm_f1",
            "y_data_key": "norm_x2",
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
            "description": "Relationship between $f_2$ and $x_1$",
            "x_label": "Normalized $f_2$",
            "y_label": "Normalized $x_1$",
            "x_data_key": "norm_f2",
            "y_data_key": "norm_x1",
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
            "description": "Relationship between $f_2$ and $x_2$",
            "x_label": "Normalized $f_2$",
            "y_label": "Normalized $x_2$",
            "x_data_key": "norm_f2",
            "y_data_key": "norm_x2",
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
            "description": "3D: $f_1$, $f_2$ and $x_1$ with interpolation",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
            "z_label": "$x_1$",
            "x_data_key": "norm_f1",
            "y_data_key": "norm_f2",
            "z_data_key": "norm_x1",
            "surface_source_key": "interpolations_2d",
            "surface_relationship_key": "f1f2_vs_x1",
            "name": "Original Points",
            "color": "#1f77b4",
            "marker_size": 5,
        },
        (6, 2): {
            "type": "scatter3d",
            "title": "3D: $f_1$, $f_2$, $x_2$",
            "description": "3D: $f_1$, $f_2$ and $x_2$ with interpolation",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
            "z_label": "$x_2$",
            "x_data_key": "norm_f1",
            "y_data_key": "norm_f2",
            "z_data_key": "norm_x2",
            "surface_source_key": "interpolations_2d",
            "surface_relationship_key": "f1f2_vs_x2",
            "name": "Original Points",
            "color": "#1f77b4",
            "marker_size": 5,
        },
    }

    _FIGURE_LAYOUT_CONFIG = {
        "title_text": "Enhanced Pareto Optimization Dashboard",
        "title_x": 0.5,
        "title_font_size": 24,
        # The height is increased to give more room for plots
        "height": 2200,
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

    def plot(self, data: Any):
        """
        Generates and displays a comprehensive Pareto optimization dashboard.

        Args:
            dto (ParetoVisualizationDTO): Data Transfer Object containing all
                                          pre-processed data for visualization.
        """

        prepared_data = ParetoDataService().prepare_dataset(data)

        dto = ParetoVisualizationMapper().map_to_dto(prepared_data)

        fig = self._create_figure_layout()
        self._add_all_subplots_from_config(fig, dto)

        fig.show()

    def _create_figure_layout(self) -> go.Figure:
        """
        Creates and configures the main figure layout for the dashboard.
        """
        rows = max(r for r, c in self._SUBPLOT_CONFIG.keys())
        cols = 2  # Fixed to 2 columns

        specs = [[None for _ in range(cols)] for _ in range(rows)]
        subplot_titles = [None] * (rows * cols)

        for (r, c), config in self._SUBPLOT_CONFIG.items():
            if 1 <= r <= rows and 1 <= c <= cols:
                # Ensure the specs are updated with the correct subplot type
                specs[r - 1][c - 1] = {"type": config["type"]}
                # Update the subplot titles to match the new grid
                subplot_titles[(r - 1) * cols + (c - 1)] = config["title"]

        # Adjusted horizontal and vertical spacing to give more room to the plots.
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,  # Reduced spacing
            vertical_spacing=0.05,  # Reduced spacing
            # Adjust column and row widths to fit the new layout
            column_widths=[0.5, 0.5],
            # All rows have the same height now for a more uniform grid.
            row_heights=[1 / rows] * rows,
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

    def _add_all_subplots_from_config(
        self, fig: go.Figure, dto: ParetoVisualizationDTO
    ) -> None:
        """
        Iterates through _SUBPLOT_CONFIG and adds traces dynamically.
        """
        for (row, col), config in self._SUBPLOT_CONFIG.items():
            plot_type = config.get("type")
            description = config.get("description", "")

            if plot_type == "scatter":
                if "data_key" in config:  # For pareto_set/front (2D array, indexed)
                    data_source = getattr(dto, config["data_key"], None)
                    x_data = self._get_data_from_indexed_source(
                        data_source, config.get("data_mapping", {}).get("x")
                    )
                    y_data = self._get_data_from_indexed_source(
                        data_source, config.get("data_mapping", {}).get("y")
                    )
                else:  # For normalized X/F scatter plots (direct DTO attributes)
                    x_data = getattr(dto, config.get("x_data_key"), None)
                    y_data = getattr(dto, config.get("y_data_key"), None)

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
                if (
                    "interpolation_source_key" in config
                    and "interpolation_relationship_key" in config
                ):
                    # Get the top-level interpolation dictionary (e.g., dto.interpolations_1d)
                    top_level_interpolations = getattr(
                        dto, config["interpolation_source_key"], None
                    )
                    # Get the specific relationship's interpolation methods dictionary (e.g., dto.interpolations_1d["f1_vs_f2"])
                    interpolation_methods_dict = (
                        top_level_interpolations.get(
                            config["interpolation_relationship_key"]
                        )
                        if top_level_interpolations
                        else None
                    )

                    if interpolation_methods_dict:
                        self._add_interpolation_traces(
                            fig,
                            row,
                            col,
                            interpolation_methods_dict,
                            self._INTERPOLATION_COLORS,
                            config,  # Pass the subplot config
                        )

            elif plot_type == "parcoords":
                parcoords_data = getattr(dto, config.get("data_key"), None)
                if parcoords_data is None or parcoords_data.size == 0:
                    print(
                        f"Warning: Missing or empty data for parcoords plot at ({row}, {col}). Skipping."
                    )
                    continue
                self._add_parcoords_plot(fig, row, col, parcoords_data, config)

            elif plot_type == "scatter3d":
                x_data = getattr(dto, config.get("x_data_key"), None)
                y_data = getattr(dto, config.get("y_data_key"), None)
                z_data = getattr(dto, config.get("z_data_key"), None)

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
                if (
                    "surface_source_key" in config
                    and "surface_relationship_key" in config
                ):
                    # Get the top-level interpolation dictionary (e.g., dto.interpolations_2d)
                    top_level_interpolations = getattr(
                        dto, config["surface_source_key"], None
                    )
                    # Get the specific relationship's interpolation methods dictionary (e.g., dto.interpolations_2d["f1f2_vs_x1"])
                    surface_methods_dict = (
                        top_level_interpolations.get(config["surface_relationship_key"])
                        if top_level_interpolations
                        else None
                    )

                    if surface_methods_dict:
                        self._add_3d_surface_traces(
                            fig,
                            row,
                            col,
                            surface_methods_dict,
                            self._INTERPOLATION_COLORS,
                            config,  # Pass the subplot config
                        )
            else:
                print(
                    f"Warning: Unsupported plot type '{plot_type}' for subplot ({row}, {col})."
                )
                continue

    def _get_data_from_indexed_source(
        self, data_source: np.ndarray | tuple, index: int | None
    ) -> np.ndarray | None:
        """
        Helper to retrieve data from a numpy array or tuple using an integer index.
        Used for pareto_set/front.
        """
        if data_source is None or index is None:
            return None

        if isinstance(data_source, np.ndarray):
            if data_source.ndim > 1:
                return data_source[:, index]
            else:  # If it's a 1D array (e.g., if we ever use this for a single normalized array), and index is 0, return itself
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
                # Add hover template for detailed info on hover
                hovertemplate=f"{config['x_label']}: %{{x:.4f}}<br>{config['y_label']}: %{{y:.4f}}<extra>{config['name']}</extra>",
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
                    color=data[:, 2]  # Example: Use the third column (f1) for color
                    if data.shape[1] > 2
                    else data[:, 0],  # Fallback to first column
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
        interpolation_methods_dict: dict[
            str, tuple[np.ndarray, np.ndarray]
        ],  # This is now the inner dict of methods
        interpolation_colors: dict,
        config: dict,  # New parameter to access subplot config
    ) -> None:
        """
        Adds interpolation lines for a scatter plot, iterating through all methods in the dict.
        """
        for method_name, (x_grid, y_grid) in interpolation_methods_dict.items():
            show_legend_item = method_name not in self._added_legend_items
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    line=dict(
                        color=interpolation_colors.get(method_name, "#000000"),
                        width=2.5,
                        dash="solid",  # Keep solid unless a dash pattern is explicitly desired per method
                    ),
                    name=method_name,
                    legendgroup=method_name,
                    showlegend=show_legend_item,
                    # Add hover template for lines
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
                # Add hover template for 3D points
                hovertemplate=f"{config['x_label']}: %{{x:.4f}}<br>{config['y_label']}: %{{y:.4f}}<br>{config['z_label']}: %{{z:.4f}}<extra>{config['name']}</extra>",
            ),
            row=row,
            col=col,
        )
        if show_legend_points:
            self._added_legend_items.add(config.get("name", "Original Points"))

        # Update scene to set a default camera view and a 'data' aspect mode to stretch the plot.
        fig.update_scenes(
            xaxis_title_text=config["x_label"],
            yaxis_title_text=config["y_label"],
            zaxis_title_text=config["z_label"],
            # Changed aspectmode from 'cube' to 'data' to stretch plots to fit data ranges.
            aspectmode="data",
            row=row,
            col=col,
            # Set a default camera position for a consistent view
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
        )

    def _add_3d_surface_traces(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        surface_methods_dict: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ],  # This is now the inner dict of methods
        interpolation_colors: dict,
        config: dict,  # New parameter to access subplot config
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
                        [0, interpolation_colors.get(method_name, "#000000")],
                        [1, interpolation_colors.get(method_name, "#000000")],
                    ],
                    opacity=0.7,
                    name=method_name,
                    showlegend=show_in_legend,
                    showscale=False,
                    legendgroup=method_name,
                    # Add hover template for surfaces
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

    def _add_description(self, fig: go.Figure, row: int, col: int, text: str) -> None:
        """
        Adds a description text annotation below a subplot.
        """
        # Ensure row and col indices are valid
        if (
            row not in self._DESCRIPTION_POSITIONS["row"]
            or col not in self._DESCRIPTION_POSITIONS["col"]
        ):
            print(
                f"Warning: No description position configured for subplot ({row}, {col})."
            )
            return

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
