from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


# Re-define ParetoVisualizationDTO here for a complete, runnable example
@dataclass
class ParetoVisualizationDTO:
    pareto_set: np.ndarray
    pareto_front: np.ndarray
    normalized_decision_space: Tuple[np.ndarray, np.ndarray]
    normalized_objective_space: Tuple[np.ndarray, np.ndarray]
    parallel_coordinates_data: np.ndarray

    f1_relationships: Dict[str, Any]
    f2_relationships: Dict[str, Any]
    x1_x2_relationship: Dict[str, Any]

    multivariate_interpolations: Dict[str, Any]

    norm_f1: np.ndarray
    norm_f2: np.ndarray
    norm_x1: np.ndarray
    norm_x2: np.ndarray


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

        # Extract normalized components for direct access
        norm_f1 = dataset.normalized["f1"]
        norm_f2 = dataset.normalized["f2"]
        norm_x1 = dataset.normalized["x1"]
        norm_x2 = dataset.normalized["x2"]

        return ParetoVisualizationDTO(
            pareto_set=dataset.pareto_set,
            pareto_front=dataset.pareto_front,
            normalized_decision_space=(
                norm_x1,
                norm_x2,
            ),  # Still pass tuples for general use
            normalized_objective_space=(
                norm_f1,
                norm_f2,
            ),  # Still pass tuples for general use
            parallel_coordinates_data=np.hstack(
                [
                    norm_x1.reshape(-1, 1),
                    norm_x2.reshape(-1, 1),
                    norm_f1.reshape(-1, 1),
                    norm_f2.reshape(-1, 1),
                ]
            ),
            # Updated relationships dictionaries to use direct keys
            f1_relationships={
                "f1_vs_f2": dataset.interpolations_1d["f1_vs_f2"],
                "f1_vs_x1": dataset.interpolations_1d["f1_vs_x1"],
                "f1_vs_x2": dataset.interpolations_1d["f1_vs_x2"],
                "norm_f1": norm_f1,  # Direct access to f1
                "norm_f2": norm_f2,  # Direct access to f2 (used in the visualizer for f2 vs f1 mapping)
                "norm_x1": norm_x1,  # Direct access to x1
                "norm_x2": norm_x2,  # Direct access to x2
            },
            f2_relationships={
                "f2_vs_x1": dataset.interpolations_1d["f2_vs_x1"],
                "f2_vs_x2": dataset.interpolations_1d["f2_vs_x2"],
                "norm_f2": norm_f2,  # Direct access to f2
                "norm_x1": norm_x1,  # Direct access to x1
                "norm_x2": norm_x2,  # Direct access to x2
            },
            x1_x2_relationship={
                "x1": norm_x1,  # Direct access to x1
                "x2": norm_x2,  # Direct access to x2
                "interpolations": dataset.interpolations_1d["x1_vs_x2"],
            },
            multivariate_interpolations=dataset.interpolations_2d,
            # Populate the new top-level normalized fields
            norm_f1=norm_f1,
            norm_f2=norm_f2,
            norm_x1=norm_x1,
            norm_x2=norm_x2,
        )

    def _validate_dataset(self, dataset: Any):
        """Validate required data exists in the dataset"""
        if dataset.pareto_set is None:
            raise ValueError("Missing pareto_set in dataset")
        if dataset.pareto_front is None:
            raise ValueError("Missing pareto_front in dataset")

        for key in ["f1", "f2", "x1", "x2"]:
            if dataset.normalized[key] is None:
                raise ValueError(f"Missing normalized {key} in dataset")

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

        required_2d_keys = ["f1f2_vs_x1", "f1f2_vs_x2"]
        for key in required_2d_keys:
            if key not in dataset.interpolations_2d:
                raise ValueError(f"Missing multivariate interpolation key: {key}")
