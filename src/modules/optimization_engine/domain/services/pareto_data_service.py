from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# These imports are for ParetoDataService, not for the InterpolatedResult types themselves
from ...infrastructure.inverse_decision_mappers.splines.monomials.cubic import (
    CubicSplineInverseDecisionMapper,
)
from ...infrastructure.inverse_decision_mappers.splines.monomials.linear import (
    LinearInverseDecisionMapper,
)
from ...infrastructure.inverse_decision_mappers.splines.monomials.pchip import (
    PchipInverseDecisionMapper,
)
from ...infrastructure.inverse_decision_mappers.splines.monomials.quadratic import (
    QuadraticInverseDecisionMapper,
)
from ...infrastructure.inverse_decision_mappers.splines.monomials.rbf import (
    RBFInverseDecisionMapper,
)
from ...infrastructure.inverse_decision_mappers.splines.multinomials.linear import (
    LinearNDInverseDecisionMapper,
)
from ...infrastructure.inverse_decision_mappers.splines.multinomials.nearest_neighbors import (
    NearestNDInverseDecisionMapper,
)
from ..generation.interfaces.base_archiver import BaseParetoArchiver
from ..interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)

# --- Interpolated1DResult and Interpolated2DResult dataclasses are REMOVED ---
# They will now be replaced by direct tuples in the ParetoDataset and ParetoDataService.


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
    # Now stores tuples of numpy arrays
    interpolations_1d: dict[str, dict[str, Tuple[np.ndarray, np.ndarray]]] = (
        field(  # Changed type hint
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
    )

    def get_1d_interpolated_data(
        self, relationship: str, method: str
    ) -> Tuple[np.ndarray, np.ndarray]:  # Changed return type hint
        """
        Retrieves the specific 1D interpolated data for a given relationship and method.

        Args:
            relationship (str): The name of the relationship (e.g., "f1_vs_f2").
            method (str): The name of the interpolation method (e.g., "Pchip", "Cubic Spline").

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the x and y interpolated data arrays.

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
    # Now stores tuples of numpy arrays
    interpolations_2d: dict[
        str, dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = field(  # Changed type hint
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Changed return type hint
        """
        Retrieves the specific 2D interpolated data for a given relationship and method.

        Args:
            relationship (str): The name of the relationship (e.g., "f1f2_vs_x1").
            method (str): The name of the mapping method (e.g., "Linear ND", "Nearest Neighbor").

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the x_grid_1, x_grid_2, and z_values arrays.

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
    Focuses on data transformation without visualization-specific logic.
    """

    # Map method names to the new custom Inverse Decision Mapper classes
    INTERPOLATION_METHODS_1D: dict[str, dict[str, Any]] = {
        "Pchip": {"min_points": 2, "class": PchipInverseDecisionMapper},
        "Cubic Spline": {"min_points": 4, "class": CubicSplineInverseDecisionMapper},
        "Linear": {"min_points": 2, "class": LinearInverseDecisionMapper},
        "Quadratic": {"min_points": 3, "class": QuadraticInverseDecisionMapper},
        "RBF": {
            "min_points": 1,
            "class": RBFInverseDecisionMapper,
            "requires_2d_input_for_fit": True,
        },
    }

    INTERPOLATION_METHODS_ND: dict[str, dict[str, Any]] = {
        "Nearest Neighbor": {"min_points": 1, "class": NearestNDInverseDecisionMapper},
        "Linear ND": {
            "min_points": 3,
            "class": LinearNDInverseDecisionMapper,
            "add_jitter_for_fit": True,
        },
    }

    # Define standard number of points for interpolation for plotting purposes
    # These could be made configurable or dynamic based on data density if needed.
    _NUM_INTERPOLATION_POINTS_1D = 100
    _NUM_INTERPOLATION_POINTS_2D_GRID = 50  # For each dimension of the 2D grid

    def __init__(self, archiver: BaseParetoArchiver):
        self.archiver = archiver

    def prepare_dataset(self, data_identifier: str | Path) -> ParetoDataset:
        """Prepare complete Pareto dataset with original, normalized, and interpolated data"""
        # Load and validate Pareto data
        loaded_result = self.archiver.load(data_identifier)

        if not hasattr(loaded_result, "pareto_set") or not hasattr(
            loaded_result, "pareto_front"
        ):
            raise ValueError(
                "Archiver did not return valid Pareto data (missing pareto_set or pareto_front)."
            )

        # Ensure loaded data is not empty
        if loaded_result.pareto_set.size == 0 or loaded_result.pareto_front.size == 0:
            raise ValueError(
                "Loaded Pareto data (pareto_set or pareto_front) is empty."
            )

        # Initialize dataset container
        dataset = ParetoDataset()
        dataset.pareto_set = loaded_result.pareto_set
        dataset.pareto_front = loaded_result.pareto_front

        # Compute and store normalized values, now directly from pareto_set/front
        if dataset.pareto_front.shape[1] < 2:
            raise ValueError("pareto_front must have at least 2 columns for f1 and f2.")
        if dataset.pareto_set.shape[1] < 2:
            raise ValueError("pareto_set must have at least 2 columns for x1 and x2.")

        dataset.normalized["f1"] = self._normalize_array(dataset.pareto_front[:, 0])
        dataset.normalized["f2"] = self._normalize_array(dataset.pareto_front[:, 1])
        dataset.normalized["x1"] = self._normalize_array(dataset.pareto_set[:, 0])
        dataset.normalized["x2"] = self._normalize_array(dataset.pareto_set[:, 1])

        # Compute all interpolations
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

    def _compute_all_1d_interpolations(self, dataset: ParetoDataset):
        """Compute all 1D inverse decision mappers and their interpolated data for various relationships."""
        norm = dataset.normalized

        # Get unique points for stable mapping (and ensure sorted for 1D methods)
        # np.unique sorts the array, we get indices to re-align dependent variables

        # For relationships where f1 is the independent variable (objective_space_points)
        sorted_indices_f1 = np.argsort(norm["f1"])
        unique_f1, unique_f1_idx = np.unique(
            norm["f1"][sorted_indices_f1], return_index=True
        )
        f2_for_f1 = norm["f2"][sorted_indices_f1][unique_f1_idx]
        x1_for_f1 = norm["x1"][sorted_indices_f1][unique_f1_idx]
        x2_for_f1 = norm["x2"][sorted_indices_f1][unique_f1_idx]

        # For relationships where x1 is the independent variable
        sorted_indices_x1 = np.argsort(norm["x1"])
        unique_x1, unique_x1_idx = np.unique(
            norm["x1"][sorted_indices_x1], return_index=True
        )
        x2_for_x1 = norm["x2"][sorted_indices_x1][unique_x1_idx]

        # For relationships where f2 is the independent variable
        sorted_indices_f2 = np.argsort(norm["f2"])
        unique_f2, unique_f2_idx = np.unique(
            norm["f2"][sorted_indices_f2], return_index=True
        )
        x1_for_f2 = norm["x1"][sorted_indices_f2][unique_f2_idx]
        x2_for_f2 = norm["x2"][sorted_indices_f2][unique_f2_idx]

        # Fit mappers, predict data, and store results
        dataset.interpolations_1d["f1_vs_f2"] = self._compute_1d_interpolated_results(
            unique_f1, f2_for_f1
        )
        dataset.interpolations_1d["f1_vs_x1"] = self._compute_1d_interpolated_results(
            unique_f1, x1_for_f1
        )
        dataset.interpolations_1d["f1_vs_x2"] = self._compute_1d_interpolated_results(
            unique_f1, x2_for_f1
        )
        dataset.interpolations_1d["x1_vs_x2"] = self._compute_1d_interpolated_results(
            unique_x1, x2_for_x1
        )
        dataset.interpolations_1d["f2_vs_x1"] = self._compute_1d_interpolated_results(
            unique_f2, x1_for_f2
        )
        dataset.interpolations_1d["f2_vs_x2"] = self._compute_1d_interpolated_results(
            unique_f2, x2_for_f2
        )

    def _compute_1d_interpolated_results(
        self, x_independent: np.ndarray, y_dependent: np.ndarray
    ) -> dict[str, Tuple[np.ndarray, np.ndarray]]:  # Changed return type hint
        """
        Fits 1D inverse decision mappers, performs prediction, and returns results as tuples.
        Returns a dictionary of method_name -> Tuple[np.ndarray, np.ndarray].
        """
        results_map = {}

        # Define the range for interpolation points
        if x_independent.size > 0:
            x_min, x_max = x_independent.min(), x_independent.max()
            # If min and max are the same (e.g., single unique point), create a small range around it
            if x_min == x_max:
                x_interpolation_range = np.array(
                    [x_min]
                )  # Just interpolate at the point itself
            else:
                x_interpolation_range = np.linspace(
                    x_min, x_max, self._NUM_INTERPOLATION_POINTS_1D
                )
        else:
            x_interpolation_range = np.array([])

        for method_name, method_info in self.INTERPOLATION_METHODS_1D.items():
            if len(x_independent) < method_info["min_points"]:
                continue

            try:
                mapper_class = method_info["class"]
                mapper_instance: BaseInverseDecisionMapper = mapper_class()

                # Prepare input if the mapper requires 2D for its independent variable (e.g., RBF)
                x_prepared_for_fit = x_independent
                if method_info.get("requires_2d_input_for_fit", False):
                    x_prepared_for_fit = x_independent.reshape(-1, 1)

                # Fit the mapper
                mapper_instance.fit(x_prepared_for_fit, y_dependent)

                # Predict the interpolated values
                if x_interpolation_range.size > 0:
                    interpolated_y_values = mapper_instance.predict(
                        x_interpolation_range
                    )
                else:
                    interpolated_y_values = np.array([])

                # Store the interpolated data as a tuple
                results_map[method_name] = (
                    x_interpolation_range,
                    interpolated_y_values,
                )  # Changed to tuple
            except Exception as e:
                print(
                    f"1D Inverse Decision Mapper instantiation/fit/predict failed for {method_name}: {str(e)}"
                )
                continue

        return results_map

    def _compute_all_2d_interpolations(self, dataset: ParetoDataset):
        """Compute all 2D multivariate inverse decision mappers and their interpolated data."""
        # For 2D mappers, the independent variables (objective_space_points) will be [f1, f2]
        X_input_independent = np.column_stack(
            (dataset.normalized["f1"], dataset.normalized["f2"])
        )

        # Fit mappers, predict data, and store results
        dataset.interpolations_2d["f1f2_vs_x1"] = self._compute_2d_interpolated_results(
            X_input_independent,
            dataset.normalized[
                "x1"
            ],  # x1 is the dependent variable (decision_space_points)
        )
        dataset.interpolations_2d["f1f2_vs_x2"] = self._compute_2d_interpolated_results(
            X_input_independent,
            dataset.normalized["x2"],  # x2 is the dependent variable
        )

    def _compute_2d_interpolated_results(
        self, X_independent: np.ndarray, y_dependent: np.ndarray
    ) -> dict[
        str, Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:  # Changed return type hint
        """
        Fits 2D inverse decision mappers, performs prediction, and returns results as tuples.
        Returns a dictionary of method_name -> Tuple[np.ndarray, np.ndarray, np.ndarray].
        """
        results_map = {}

        # Define the 2D grid for interpolation points
        if X_independent.shape[0] > 0:
            f1_min, f1_max = X_independent[:, 0].min(), X_independent[:, 0].max()
            f2_min, f2_max = X_independent[:, 1].min(), X_independent[:, 1].max()

            # Handle cases where min == max for a dimension
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

            # If only one point, make grid for that single point
            if grid_f1.size == 1 and grid_f2.size == 1:
                mesh_f1, mesh_f2 = grid_f1, grid_f2
            else:
                mesh_f1, mesh_f2 = np.meshgrid(grid_f1, grid_f2)

            # Prepare points for prediction (flattened grid)
            points_for_prediction = np.column_stack((mesh_f1.ravel(), mesh_f2.ravel()))
        else:
            # No data points, no grid or predictions
            mesh_f1, mesh_f2, points_for_prediction = (
                np.array([]),
                np.array([]),
                np.array([]),
            )

        for method_name, method_info in self.INTERPOLATION_METHODS_ND.items():
            if len(X_independent) < method_info["min_points"]:
                continue

            try:
                mapper_class = method_info["class"]
                mapper_instance: BaseInverseDecisionMapper = mapper_class()

                # Add jitter if needed for stability for certain ND methods (e.g., Linear ND)
                X_prepared_for_fit = X_independent
                if method_info.get("add_jitter_for_fit", False):
                    # Only add jitter if there's actual data to jitter
                    if X_independent.shape[0] > 0:
                        jitter = np.random.normal(0, 1e-10, size=X_independent.shape)
                        X_prepared_for_fit = X_independent + jitter

                # Fit the mapper
                mapper_instance.fit(X_prepared_for_fit, y_dependent)

                # Predict the interpolated values
                if points_for_prediction.shape[0] > 0:
                    interpolated_z_values_flat = mapper_instance.predict(
                        points_for_prediction
                    )
                    # Reshape the predicted values back to the grid shape
                    interpolated_z_values = interpolated_z_values_flat.reshape(
                        mesh_f1.shape
                    )
                else:
                    interpolated_z_values = np.array([])

                # Store the interpolated data as a tuple
                results_map[method_name] = (
                    mesh_f1,
                    mesh_f2,
                    interpolated_z_values,
                )  # Changed to tuple
            except Exception as e:
                print(
                    f"2D Inverse Decision Mapper instantiation/fit/predict failed for {method_name}: {str(e)}"
                )
                continue

        return results_map
