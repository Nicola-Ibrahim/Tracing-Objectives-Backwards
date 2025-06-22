from dataclasses import dataclass
from typing import Any

import numpy as np


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

    def map_to_dto(self, dataset: Any) -> ParetoVisualizationDTO:
        """
        Transform dataset into visualization DTO
        """
        # Orchestrate validation using dedicated methods
        self._validate_core_data(dataset)
        self._validate_normalized_data(dataset)
        self._validate_1d_interpolations(dataset)
        self._validate_2d_interpolations(dataset)

        # Extract normalized components for direct access in DTO
        # These come from the 'normalized' dictionary in ParetoDataset
        norm_x1 = dataset.normalized["x1"]
        norm_x2 = dataset.normalized["x2"]
        norm_f1 = dataset.normalized["f1"]
        norm_f2 = dataset.normalized["f2"]

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

    def _validate_core_data(self, dataset: Any):
        """Validates the core Pareto set and front arrays, including their dimensions."""
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

    def _validate_normalized_data(self, dataset: Any):
        """Validates the normalized data dictionary."""
        required_norm_keys = ["f1", "f2", "x1", "x2"]
        if not isinstance(dataset.normalized, dict):
            raise TypeError("Normalized data 'dataset.normalized' is not a dictionary.")
        for key in required_norm_keys:
            if key not in dataset.normalized or not isinstance(
                dataset.normalized[key], np.ndarray
            ):
                raise ValueError(
                    f"Missing or invalid normalized '{key}' in dataset (must be np.ndarray)."
                )
            if dataset.normalized[key].ndim != 1:
                raise ValueError(f"Normalized '{key}' should be a 1D array.")
            if dataset.normalized[key].shape[0] != dataset.pareto_set.shape[0]:
                raise ValueError(
                    f"Number of samples in normalized '{key}' ({dataset.normalized[key].shape[0]}) does not match 'pareto_set' ({dataset.pareto_set.shape[0]})."
                )

    def _validate_1d_interpolations(self, dataset: Any):
        """Validates the 1D interpolations dictionary."""
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

    def _validate_2d_interpolations(self, dataset: Any):
        """Validates the 2D interpolations dictionary."""
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
