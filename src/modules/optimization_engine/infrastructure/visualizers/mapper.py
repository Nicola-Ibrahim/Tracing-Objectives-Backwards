from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ParetoVisualizationDTO:
    """
    Data Transfer Object for Pareto visualization data.
    Provides a strict interface for the visualizer with validated data structure.
    """

    pareto_set: np.ndarray
    pareto_front: np.ndarray
    normalized_decision_space: tuple[np.ndarray, np.ndarray]
    normalized_objective_space: tuple[np.ndarray, np.ndarray]
    parallel_coordinates_data: np.ndarray
    f1_relationships: dict[str, np.ndarray]
    f2_relationships: dict[str, np.ndarray]
    x1_x2_relationship: dict[str, np.ndarray]
    multivariate_interpolations: dict[str, np.ndarray]


class ParetoVisualizationMapper:
    """
    Maps ParetoDataset to a structured DTO suitable for visualization.
    Performs data validation and transformation.
    """

    def map_to_dto(self, dataset: Any) -> ParetoVisualizationDTO:
        """
        Transform dataset into visualization DTO
        """
        self._validate_dataset(dataset)

        return ParetoVisualizationDTO(
            pareto_set=dataset.pareto_set,
            pareto_front=dataset.pareto_front,
            normalized_decision_space=(
                dataset.normalized["x1"],
                dataset.normalized["x2"],
            ),
            normalized_objective_space=(
                dataset.normalized["f1"],
                dataset.normalized["f2"],
            ),
            parallel_coordinates_data=np.hstack(
                [
                    dataset.normalized["x1"].reshape(-1, 1),
                    dataset.normalized["x2"].reshape(-1, 1),
                    dataset.normalized["f1"].reshape(-1, 1),
                    dataset.normalized["f2"].reshape(-1, 1),
                ]
            ),
            f1_relationships={
                "f1_vs_f2": dataset.interpolations_1d["f1_vs_f2"],
                "f1_vs_x1": dataset.interpolations_1d["f1_vs_x1"],
                "f1_vs_x2": dataset.interpolations_1d["f1_vs_x2"],
                "norm_f1": dataset.normalized["f1"],
            },
            f2_relationships={
                "f2_vs_x1": dataset.interpolations_1d["f2_vs_x1"],
                "f2_vs_x2": dataset.interpolations_1d["f2_vs_x2"],
                "norm_f2": dataset.normalized["f2"],
            },
            x1_x2_relationship={
                "x1": dataset.normalized["x1"],
                "x2": dataset.normalized["x2"],
                "interpolations": dataset.interpolations_1d["x1_vs_x2"],
            },
            multivariate_interpolations=dataset.interpolations_2d,
        )

    def _validate_dataset(self, dataset: Any):
        """Validate required data exists in the dataset"""
        # Validate core data
        if dataset.pareto_set is None:
            raise ValueError("Missing pareto_set in dataset")
        if dataset.pareto_front is None:
            raise ValueError("Missing pareto_front in dataset")

        # Validate normalized data
        for key in ["f1", "f2", "x1", "x2"]:
            if dataset.normalized[key] is None:
                raise ValueError(f"Missing normalized {key} in dataset")

        # Validate 1D interpolations
        required_1d_keys = [
            "f1_vs_f2",
            "f1_vs_x1",
            "f1_vs_x2",
            "x1_vs_x2",
            "f2_vs_x1",
            "f2_vs_x2",
        ]
        for key in required_1d_keys:
            if key not in dataset.interpolations_1d:
                raise ValueError(f"Missing interpolation key: {key}")

        # Validate 2D interpolations
        required_2d_keys = ["f1f2_vs_x1", "f1f2_vs_x2"]
        for key in required_2d_keys:
            if key not in dataset.interpolations_2d:
                raise ValueError(f"Missing multivariate interpolation key: {key}")
