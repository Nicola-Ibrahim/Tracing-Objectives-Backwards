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


class ParetoDataService:
    """
    Application Service responsible for orchestrating the retrieval of raw Pareto data,
    its preparation, and interpolation computation for visualization.
    """

    # Define interpolation methods with their minimum required points
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

    def provide_visualization_data(
        self, data_identifier: str | Path
    ) -> tuple[dict, dict]:
        loaded_result = self.archiver.load(data_identifier)

        if not hasattr(loaded_result, "pareto_set") or not hasattr(
            loaded_result, "pareto_front"
        ):
            raise ValueError("Archiver did not return valid Pareto data")

        pareto_set = loaded_result.pareto_set
        pareto_front = loaded_result.pareto_front

        # Normalize all data once
        norm_data = self._normalize_all_data(pareto_set, pareto_front)

        # Prepare data and compute interpolations
        f1_rel_data = self._prepare_f1_relationships_data(norm_data)
        interp_data = self._prepare_interpolation_data(norm_data)

        # Add multivariate interpolations to interp_data
        interp_data["multivariate_interpolations"] = self._prepare_multivariate_data(
            norm_data
        )

        return f1_rel_data, interp_data

    def _normalize_all_data(self, pareto_set, pareto_front):
        """Normalize all data and return in a dictionary"""
        f1_orig = pareto_front[:, 0]
        f2_orig = pareto_front[:, 1]
        x1_orig = pareto_set[:, 0]
        x2_orig = pareto_set[:, 1]

        return {
            "f1_orig": f1_orig,
            "f2_orig": f2_orig,
            "x1_orig": x1_orig,
            "x2_orig": x2_orig,
            "norm_f1": self._normalize_array(f1_orig),
            "norm_f2": self._normalize_array(f2_orig),
            "norm_x1": self._normalize_array(x1_orig),
            "norm_x2": self._normalize_array(x2_orig),
        }

    def _normalize_array(self, data_array: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler()
        reshaped_data = (
            data_array.reshape(-1, 1) if data_array.ndim == 1 else data_array
        )
        normalized_data = scaler.fit_transform(reshaped_data)
        return normalized_data.flatten() if data_array.ndim == 1 else normalized_data

    def _compute_1d_interpolations(
        self, x_unique, y_unique, x_range=(0, 1), num_points=100
    ):
        """Compute 1D interpolations for all methods that have enough points"""
        interpolations = {}
        x_grid = np.linspace(x_range[0], x_range[1], num_points)

        for method_name, method_info in self.INTERPOLATION_METHODS_1D.items():
            if len(x_unique) < method_info["min_points"]:
                continue  # Skip if not enough points

            try:
                # Prepare input data according to method requirements
                if method_info.get("requires_2d", False):
                    x_prepared = x_unique.reshape(-1, 1)
                else:
                    x_prepared = x_unique

                # Create interpolation function
                interp_func = method_info["function"](x_prepared, y_unique)

                # Evaluate interpolation
                if method_info.get("requires_2d", False):
                    eval_points = x_grid.reshape(-1, 1)
                    y_grid = interp_func(eval_points).flatten()
                else:
                    y_grid = interp_func(x_grid)

                interpolations[method_name] = (x_grid, y_grid)
            except Exception as e:
                print(f"1D Interpolation failed for {method_name}: {str(e)}")
                continue

        return interpolations

    def _compute_2d_interpolations(
        self, X, y, x_range=(0, 1), y_range=(0, 1), grid_size=20
    ):
        """Compute 2D interpolations for multivariate relationships"""
        interpolations = {}

        # Create grid for evaluation
        x_grid = np.linspace(x_range[0], x_range[1], grid_size)
        y_grid = np.linspace(y_range[0], y_range[1], grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

        for method_name, method_info in self.INTERPOLATION_METHODS_ND.items():
            if len(X) < method_info["min_points"]:
                continue  # Skip if not enough points

            try:
                # Add jitter if specified to prevent Qhull errors
                if method_info.get("add_jitter", False):
                    jitter = np.random.normal(0, 1e-10, size=X.shape)
                    X_prepared = X + jitter
                else:
                    X_prepared = X

                # Create interpolation function
                interp_func = method_info["function"](X_prepared, y)

                # Evaluate interpolation on grid
                Z_grid = interp_func(grid_points)
                Z_grid = Z_grid.reshape(X_grid.shape)

                interpolations[method_name] = (X_grid, Y_grid, Z_grid)
            except Exception as e:
                print(f"2D Interpolation failed for {method_name}: {str(e)}")
                continue

        return interpolations

    def _prepare_f1_relationships_data(self, norm_data: dict) -> dict:
        # Extract normalized data
        norm_f1 = norm_data["norm_f1"]
        norm_f2 = norm_data["norm_f2"]
        norm_x1 = norm_data["norm_x1"]
        norm_x2 = norm_data["norm_x2"]

        # Get unique f1 points
        unique_norm_f1, unique_idx = np.unique(norm_f1, return_index=True)
        norm_f1_unique = norm_f1[unique_idx]
        norm_f2_unique = norm_f2[unique_idx]
        norm_x1_unique = norm_x1[unique_idx]
        norm_x2_unique = norm_x2[unique_idx]

        # Compute interpolations for f1 relationships
        f1_vs_f2 = self._compute_1d_interpolations(norm_f1_unique, norm_f2_unique)
        f1_vs_x1 = self._compute_1d_interpolations(norm_f1_unique, norm_x1_unique)
        f1_vs_x2 = self._compute_1d_interpolations(norm_f1_unique, norm_x2_unique)

        return {
            "f1_orig": norm_data["f1_orig"],
            "f2_orig": norm_data["f2_orig"],
            "x1_orig": norm_data["x1_orig"],
            "x2_orig": norm_data["x2_orig"],
            "norm_f1": norm_f1,
            "norm_f2": norm_f2,
            "norm_x1": norm_x1,
            "norm_x2": norm_x2,
            "interpolations": {
                "f1_vs_f2": f1_vs_f2,
                "f1_vs_x1": f1_vs_x1,
                "f1_vs_x2": f1_vs_x2,
            },
        }

    def _prepare_interpolation_data(self, norm_data: dict) -> dict:
        # Extract normalized data
        norm_x1 = norm_data["norm_x1"]
        norm_x2 = norm_data["norm_x2"]
        norm_f2 = norm_data["norm_f2"]

        # Prepare x1 vs x2 interpolation data
        unique_norm_x1, unique_idx = np.unique(norm_x1, return_index=True)
        norm_x2_unique = norm_x2[unique_idx]
        x1_vs_x2 = self._compute_1d_interpolations(unique_norm_x1, norm_x2_unique)

        # Prepare f2 relationships data
        sorted_idx = np.argsort(norm_f2)
        norm_f2_sorted = norm_f2[sorted_idx]
        norm_x1_sorted = norm_x1[sorted_idx]
        norm_x2_sorted = norm_x2[sorted_idx]

        norm_f2_unique, unique_idx = np.unique(norm_f2_sorted, return_index=True)
        norm_x1_unique = norm_x1_sorted[unique_idx]
        norm_x2_unique = norm_x2_sorted[unique_idx]

        # Compute interpolations for f2 relationships
        f2_vs_x1 = self._compute_1d_interpolations(norm_f2_unique, norm_x1_unique)
        f2_vs_x2 = self._compute_1d_interpolations(norm_f2_unique, norm_x2_unique)

        return {
            "x1_orig": norm_data["x1_orig"],
            "x2_orig": norm_data["x2_orig"],
            "f2_orig": norm_data["f2_orig"],
            "norm_x1": norm_x1,
            "norm_x2": norm_x2,
            "norm_f2": norm_f2,
            "interpolations": {
                "x1_vs_x2": x1_vs_x2,
                "f2_vs_x1": f2_vs_x1,
                "f2_vs_x2": f2_vs_x2,
            },
        }

    def _prepare_multivariate_data(self, norm_data: dict) -> dict:
        """Prepare multivariate interpolation data for (f1,f2) vs (x1,x2)"""
        # Create input matrix [f1, f2] for all points
        X_input = np.column_stack((norm_data["norm_f1"], norm_data["norm_f2"]))

        # Compute multivariate interpolations
        f1f2_vs_x1 = self._compute_2d_interpolations(X_input, norm_data["norm_x1"])
        f1f2_vs_x2 = self._compute_2d_interpolations(X_input, norm_data["norm_x2"])

        return {
            "f1f2_vs_x1": f1f2_vs_x1,
            "f1f2_vs_x2": f1f2_vs_x2,
            "input_matrix": X_input,
            "norm_f1": norm_data["norm_f1"],
            "norm_f2": norm_data["norm_f2"],
            "norm_x1": norm_data["norm_x1"],
            "norm_x2": norm_data["norm_x2"],
        }
