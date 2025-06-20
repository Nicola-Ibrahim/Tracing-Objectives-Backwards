from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
from sklearn.preprocessing import MinMaxScaler

from ..generation.interfaces.base_archiver import BaseParetoArchiver


class ParetoDataService:
    """
    Application Service responsible for orchestrating the retrieval of raw Pareto data,
    its preparation, and interpolation computation for visualization.
    """

    # Define interpolation methods with their minimum required points
    INTERPOLATION_METHODS = {
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
    }

    def __init__(self, archiver: BaseParetoArchiver):
        self.archiver = archiver

    def provide_visualization_data(
        self, data_identifier: str | Path
    ) -> tuple[dict, dict]:
        print(f"ParetoDataService: Requesting raw Pareto data for '{data_identifier}'")
        loaded_result = self.archiver.load(data_identifier)

        if not hasattr(loaded_result, "pareto_set") or not hasattr(
            loaded_result, "pareto_front"
        ):
            raise ValueError("Archiver did not return valid Pareto data")

        pareto_set = loaded_result.pareto_set
        pareto_front = loaded_result.pareto_front
        print("ParetoDataService: Raw data retrieved. Preparing visualization data...")

        # Prepare data and compute interpolations
        f1_rel_data = self._prepare_f1_relationships_data(pareto_set, pareto_front)
        interp_data = self._prepare_interpolation_data(pareto_set, pareto_front)

        return f1_rel_data, interp_data

    def _normalize_array(self, data_array: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler()
        reshaped_data = (
            data_array.reshape(-1, 1) if data_array.ndim == 1 else data_array
        )
        normalized_data = scaler.fit_transform(reshaped_data)
        return normalized_data.flatten() if data_array.ndim == 1 else normalized_data

    def _compute_interpolations(
        self, x_unique, y_unique, x_range=(0, 1), num_points=100
    ):
        """Compute interpolations for all methods that have enough points"""
        interpolations = {}
        x_grid = np.linspace(x_range[0], x_range[1], num_points)

        for method_name, method_info in self.INTERPOLATION_METHODS.items():
            if len(x_unique) < method_info["min_points"]:
                continue  # Skip if not enough points

            try:
                interp_func = method_info["function"](x_unique, y_unique)
                y_grid = interp_func(x_grid)
                interpolations[method_name] = (x_grid, y_grid)
            except Exception as e:
                print(f"Interpolation failed for {method_name}: {str(e)}")
                continue

        return interpolations

    def _prepare_f1_relationships_data(
        self, pareto_set: np.ndarray, pareto_front: np.ndarray
    ) -> dict:
        # Extract and normalize data
        f1_orig = pareto_front[:, 0]
        f2_orig = pareto_front[:, 1]
        x1_orig = pareto_set[:, 0]
        x2_orig = pareto_set[:, 1]

        norm_f1_all = self._normalize_array(f1_orig)
        norm_f2_all = self._normalize_array(f2_orig)
        norm_x1_all = self._normalize_array(x1_orig)
        norm_x2_all = self._normalize_array(x2_orig)

        # Get unique f1 points
        unique_norm_f1, unique_idx = np.unique(norm_f1_all, return_index=True)
        norm_f1_unique = norm_f1_all[unique_idx]
        norm_f2_unique = norm_f2_all[unique_idx]
        norm_x1_unique = norm_x1_all[unique_idx]
        norm_x2_unique = norm_x2_all[unique_idx]

        # Compute interpolations for f1 relationships
        f1_vs_f2 = self._compute_interpolations(norm_f1_unique, norm_f2_unique)
        f1_vs_x1 = self._compute_interpolations(norm_f1_unique, norm_x1_unique)
        f1_vs_x2 = self._compute_interpolations(norm_f1_unique, norm_x2_unique)

        return {
            "f1_orig": f1_orig,
            "f2_orig": f2_orig,
            "x1_orig": x1_orig,
            "x2_orig": x2_orig,
            "norm_f1_all": norm_f1_all,
            "norm_f2_all": norm_f2_all,
            "norm_x1_all": norm_x1_all,
            "norm_x2_all": norm_x2_all,
            "interpolations": {
                "f1_vs_f2": f1_vs_f2,
                "f1_vs_x1": f1_vs_x1,
                "f1_vs_x2": f1_vs_x2,
            },
        }

    def _prepare_interpolation_data(
        self, pareto_set: np.ndarray, pareto_front: np.ndarray
    ) -> dict:
        # Extract and normalize data
        x1_orig = pareto_set[:, 0]
        x2_orig = pareto_set[:, 1]
        f2_orig = pareto_front[:, 1]

        norm_x1_all = self._normalize_array(x1_orig)
        norm_x2_all = self._normalize_array(x2_orig)
        norm_f2_all = self._normalize_array(f2_orig)

        # Prepare x1 vs x2 interpolation data
        unique_norm_x1, unique_idx = np.unique(norm_x1_all, return_index=True)
        norm_x2_unique = norm_x2_all[unique_idx]
        x1_vs_x2 = self._compute_interpolations(unique_norm_x1, norm_x2_unique)

        # Prepare f2 relationships data
        sorted_idx = np.argsort(norm_f2_all)
        norm_f2_sorted = norm_f2_all[sorted_idx]
        norm_x1_sorted = norm_x1_all[sorted_idx]
        norm_x2_sorted = norm_x2_all[sorted_idx]

        norm_f2_unique, unique_idx = np.unique(norm_f2_sorted, return_index=True)
        norm_x1_unique = norm_x1_sorted[unique_idx]
        norm_x2_unique = norm_x2_sorted[unique_idx]

        # Compute interpolations for f2 relationships
        f2_vs_x1 = self._compute_interpolations(norm_f2_unique, norm_x1_unique)
        f2_vs_x2 = self._compute_interpolations(norm_f2_unique, norm_x2_unique)

        return {
            "x1_orig": x1_orig,
            "x2_orig": x2_orig,
            "f2_orig": f2_orig,
            "norm_x1_all": norm_x1_all,
            "norm_x2_all": norm_x2_all,
            "norm_f2_all": norm_f2_all,
            "interpolations": {
                "x1_vs_x2": x1_vs_x2,
                "f2_vs_x1": f2_vs_x1,
                "f2_vs_x2": f2_vs_x2,
            },
        }
