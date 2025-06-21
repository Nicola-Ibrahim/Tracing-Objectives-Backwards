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
    - Original values
    - Normalized values
    - Computed interpolations
    """

    # Core Pareto data
    pareto_set: np.ndarray = None
    pareto_front: np.ndarray = None

    # Original values
    original: dict[str, np.ndarray] = field(
        default_factory=lambda: {"f1": None, "f2": None, "x1": None, "x2": None}
    )

    # Normalized values (0-1 range)
    normalized: dict[str, np.ndarray] = field(
        default_factory=lambda: {"f1": None, "f2": None, "x1": None, "x2": None}
    )

    # 1D interpolations: (x_grid, y_grid) tuples
    interpolations_1d: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=lambda: {
            "f1_vs_f2": {},
            "f1_vs_x1": {},
            "f1_vs_x2": {},
            "x1_vs_x2": {},
            "f2_vs_x1": {},
            "f2_vs_x2": {},
        }
    )

    # 2D multivariate interpolations: (X_grid, Y_grid, Z_grid) tuples
    interpolations_2d: dict[
        str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = field(
        default_factory=lambda: {
            "f1f2_vs_x1": {},
            "f1f2_vs_x2": {},
        }
    )


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

        # Extract and store original values
        dataset.original["f1"] = dataset.pareto_front[:, 0]
        dataset.original["f2"] = dataset.pareto_front[:, 1]
        dataset.original["x1"] = dataset.pareto_set[:, 0]
        dataset.original["x2"] = dataset.pareto_set[:, 1]

        # Compute and store normalized values
        dataset.normalized["f1"] = self._normalize_array(dataset.original["f1"])
        dataset.normalized["f2"] = self._normalize_array(dataset.original["f2"])
        dataset.normalized["x1"] = self._normalize_array(dataset.original["x1"])
        dataset.normalized["x2"] = self._normalize_array(dataset.original["x2"])

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
