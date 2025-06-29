from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.interpolate as spi
from sklearn.preprocessing import MinMaxScaler


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
