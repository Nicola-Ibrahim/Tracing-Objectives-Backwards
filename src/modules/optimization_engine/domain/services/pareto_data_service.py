from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import (
    CubicSpline,
    LinearNDInterpolator,
    NearestNDInterpolator,
    PchipInterpolator,
    RBFInterpolator,
    interp1d,
)
from sklearn.preprocessing import MinMaxScaler

from ..generation.interfaces.base_archiver import BaseParetoArchiver


@dataclass
class ParetoDataset:
    """
    Container for all Pareto optimization data in various representations:
    - Core Pareto set and front (original values)
    - Normalized values
    - Computed interpolations

    This version removes the redundant 'original' dictionary, as original
    decision and objective values are directly accessible from pareto_set
    and pareto_front, respectively.
    """

    # --- Core Pareto Data ---
    # These are the primary sources for original decision and objective values.
    # pareto_set[:,0] is x1, pareto_set[:,1] is x2
    # pareto_front[:,0] is f1, pareto_front[:,1] is f2
    pareto_set: np.ndarray = field(
        default=None,
        metadata={
            "description": "Original decision variables (X) for Pareto optimal solutions. Expected shape (n_samples, n_decision_vars)."
        },
    )
    pareto_front: np.ndarray = field(
        default=None,
        metadata={
            "description": "Original objective function values (F) for Pareto optimal solutions. Expected shape (n_samples, n_objective_vars)."
        },
    )

    # --- Normalized Values (0-1 range) ---
    # We maintain this dictionary for flexibility, but expose properties for direct access.
    # These are 1D arrays extracted from the normalized Pareto front/set during processing.
    normalized: dict[str, np.ndarray, None] = field(
        default_factory=lambda: {"f1": None, "f2": None, "x1": None, "x2": None},
        metadata={
            "description": "Dictionary of normalized (0-1 range) decision and objective variables."
        },
    )

    @property
    def norm_f1(self) -> np.ndarray:
        """Returns the normalized values of objective function 1."""
        f1 = self.normalized.get("f1")
        if f1 is None:
            raise AttributeError("Normalized 'f1' data is not set in ParetoDataset.")
        return f1

    @property
    def norm_f2(self) -> np.ndarray:
        """Returns the normalized values of objective function 2."""
        f2 = self.normalized.get("f2")
        if f2 is None:
            raise AttributeError("Normalized 'f2' data is not set in ParetoDataset.")
        return f2

    @property
    def norm_x1(self) -> np.ndarray:
        """Returns the normalized values of decision variable 1."""
        x1 = self.normalized.get("x1")
        if x1 is None:
            raise AttributeError("Normalized 'x1' data is not set in ParetoDataset.")
        return x1

    @property
    def norm_x2(self) -> np.ndarray:
        """Returns the normalized values of decision variable 2."""
        x2 = self.normalized.get("x2")
        if x2 is None:
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
            "description": "1D interpolations for various variable relationships."
        },
    )

    def get_1d_interpolation(
        self, relationship: str, method: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves a specific 1D interpolation (x_grid, y_grid) for a given relationship and method.

        Args:
            relationship (str): The name of the relationship (e.g., "f1_vs_f2").
            method (str): The name of the interpolation method (e.g., "Pchip", "Cubic Spline").

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the x_grid and y_grid.

        Raises:
            KeyError: If the relationship or method is not found.
        """
        if relationship not in self.interpolations_1d:
            raise KeyError(f"1D interpolation relationship '{relationship}' not found.")
        if method not in self.interpolations_1d[relationship]:
            raise KeyError(
                f"Interpolation method '{method}' not found for relationship '{relationship}'."
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
            "description": "2D interpolations for multivariate relationships (surfaces)."
        },
    )

    def get_2d_interpolation(
        self, relationship: str, method: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves a specific 2D interpolation (X_grid, Y_grid, Z_grid) for a given relationship and method.

        Args:
            relationship (str): The name of the relationship (e.g., "f1f2_vs_x1").
            method (str): The name of the interpolation method (e.g., "Linear ND", "RBF").

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the X_grid, Y_grid, and Z_grid.

        Raises:
            KeyError: If the relationship or method is not found.
        """
        if relationship not in self.interpolations_2d:
            raise KeyError(f"2D interpolation relationship '{relationship}' not found.")
        if method not in self.interpolations_2d[relationship]:
            raise KeyError(
                f"Interpolation method '{method}' not found for relationship '{relationship}'."
            )
        return self.interpolations_2d[relationship][method]


class ParetoDataService:
    """
    Service for preparing Pareto optimization data in various representations.
    Focuses on data transformation without visualization-specific logic.
    """

    INTERPOLATION_METHODS_1D = {
        "Pchip": {"min_points": 2, "function": PchipInterpolator},
        "Cubic Spline": {"min_points": 4, "function": CubicSpline},
        "Linear": {
            "min_points": 2,
            "function": lambda x, y: interp1d(
                x, y, kind="linear", fill_value="extrapolate"
            ),
        },
        "Quadratic": {
            "min_points": 3,
            "function": lambda x, y: interp1d(
                x, y, kind="quadratic", fill_value="extrapolate"
            ),
        },
        "RBF": {"min_points": 1, "function": RBFInterpolator, "requires_2d": True},
    }

    INTERPOLATION_METHODS_ND = {
        "Nearest Neighbor": {"min_points": 1, "function": NearestNDInterpolator},
        "Linear ND": {
            "min_points": 3,
            "function": LinearNDInterpolator,
            "add_jitter": True,
        },
    }

    def __init__(self, archiver: BaseParetoArchiver):
        self.archiver = archiver

    def prepare_dataset(self, data_identifier: str | Path) -> ParetoDataset:
        """Prepare complete Pareto dataset with original, normalized, and interpolated data"""
        # Load and validate Pareto data
        loaded_result = self.archiver.load(data_identifier)
        if not hasattr(loaded_result, "pareto_set") or not hasattr(
            loaded_result, "pareto_front"
        ):
            raise ValueError("Archiver did not return valid Pareto data")

        # Initialize dataset container
        dataset = ParetoDataset()
        dataset.pareto_set = loaded_result.pareto_set
        dataset.pareto_front = loaded_result.pareto_front

        # Compute and store normalized values, now directly from pareto_set/front
        # Ensure that pareto_front and pareto_set have at least 2 columns
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
        scaler = MinMaxScaler()
        reshaped_data = data_array.reshape(-1, 1)
        normalized_data = scaler.fit_transform(reshaped_data)
        return normalized_data.flatten()

    def _compute_all_1d_interpolations(self, dataset: ParetoDataset):
        """Compute all 1D interpolation relationships"""
        norm = dataset.normalized

        # Get unique points for stable interpolation
        # Note: These still correctly use the *normalized* values
        unique_f1, f1_idx = np.unique(norm["f1"], return_index=True)
        unique_x1, x1_idx = np.unique(norm["x1"], return_index=True)

        # For f2 relationships, sort by f2
        f2_sorted_idx = np.argsort(norm["f2"])
        norm_f2_sorted = norm["f2"][f2_sorted_idx]
        unique_f2, f2_idx = np.unique(norm_f2_sorted, return_index=True)

        # Compute all 1D relationships
        dataset.interpolations_1d["f1_vs_f2"] = self._compute_1d_interpolation(
            unique_f1, norm["f2"][f1_idx]
        )
        dataset.interpolations_1d["f1_vs_x1"] = self._compute_1d_interpolation(
            unique_f1, norm["x1"][f1_idx]
        )
        dataset.interpolations_1d["f1_vs_x2"] = self._compute_1d_interpolation(
            unique_f1, norm["x2"][f1_idx]
        )
        dataset.interpolations_1d["x1_vs_x2"] = self._compute_1d_interpolation(
            unique_x1, norm["x2"][x1_idx]
        )
        dataset.interpolations_1d["f2_vs_x1"] = self._compute_1d_interpolation(
            unique_f2, norm["x1"][f2_sorted_idx][f2_idx]
        )
        dataset.interpolations_1d["f2_vs_x2"] = self._compute_1d_interpolation(
            unique_f2, norm["x2"][f2_sorted_idx][f2_idx]
        )

    def _compute_1d_interpolation(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Compute 1D interpolations for a relationship"""
        interpolations = {}
        x_grid = np.linspace(0, 1, 100)

        for method_name, method_info in self.INTERPOLATION_METHODS_1D.items():
            if len(x) < method_info["min_points"]:
                continue  # Skip if not enough points

            try:
                # Prepare input based on method requirements
                if method_info.get("requires_2d", False):
                    x_prepared = x.reshape(-1, 1)
                    eval_points = x_grid.reshape(-1, 1)
                else:
                    x_prepared = x
                    eval_points = x_grid

                # Create and evaluate interpolation
                interp_func = method_info["function"](x_prepared, y)
                y_grid = interp_func(eval_points)

                # Flatten if needed
                if hasattr(y_grid, "flatten"):
                    y_grid = y_grid.flatten()

                interpolations[method_name] = (x_grid, y_grid)
            except Exception as e:
                print(f"1D Interpolation failed for {method_name}: {str(e)}")
                continue

        return interpolations

    def _compute_all_2d_interpolations(self, dataset: ParetoDataset):
        """Compute all 2D multivariate interpolations"""
        # Prepare input matrix [f1, f2]
        X_input = np.column_stack((dataset.normalized["f1"], dataset.normalized["f2"]))

        # Compute relationships
        dataset.interpolations_2d["f1f2_vs_x1"] = self._compute_2d_interpolation(
            X_input, dataset.normalized["x1"]
        )
        dataset.interpolations_2d["f1f2_vs_x2"] = self._compute_2d_interpolation(
            X_input, dataset.normalized["x2"]
        )

    def _compute_2d_interpolation(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute 2D interpolations for multivariate relationships"""
        interpolations = {}
        grid_size = 20
        x_grid = np.linspace(0, 1, grid_size)
        y_grid = np.linspace(0, 1, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

        for method_name, method_info in self.INTERPOLATION_METHODS_ND.items():
            if len(X) < method_info["min_points"]:
                continue  # Skip if not enough points

            try:
                # Add jitter if needed for stability
                if method_info.get("add_jitter", False):
                    jitter = np.random.normal(0, 1e-10, size=X.shape)
                    X_prepared = X + jitter
                else:
                    X_prepared = X

                # Create and evaluate interpolation
                interp_func = method_info["function"](X_prepared, y)
                Z_grid = interp_func(grid_points).reshape(X_grid.shape)

                interpolations[method_name] = (X_grid, Y_grid, Z_grid)
            except Exception as e:
                print(f"2D Interpolation failed for {method_name}: {str(e)}")
                continue

        return interpolations
