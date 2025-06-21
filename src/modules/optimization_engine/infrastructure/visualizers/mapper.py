from typing import Any, Dict, NamedTuple, Tuple

import numpy as np


class ParetoVisualizationDTO(NamedTuple):
    """
    Data Transfer Object for Pareto visualization data.
    Provides a strict interface for the visualizer with validated data structure.
    """

    pareto_set: np.ndarray
    pareto_front: np.ndarray
    normalized_decision_space: Tuple[np.ndarray, np.ndarray]
    normalized_objective_space: Tuple[np.ndarray, np.ndarray]
    parallel_coordinates_data: np.ndarray
    f1_relationships: Dict[str, Any]
    f2_relationships: Dict[str, Any]
    x1_x2_relationship: Dict[str, Any]
    multivariate_interpolations: Dict[str, Any]


class ParetoVisualizationMapper:
    """
    Maps raw data from ParetoDataService to a structured DTO suitable for visualization.
    Performs data validation and transformation.
    """

    def __init__(self):
        """
        Initializes the mapper.
        No specific initialization required, but can be extended in the future.
        """
        pass

    def map_to_dto(self, f1_data: dict, interp_data: dict) -> ParetoVisualizationDTO:
        """
        Transform raw service data into visualization DTO
        """
        ParetoVisualizationMapper._validate_data(f1_data, interp_data)

        return ParetoVisualizationDTO(
            pareto_set=np.hstack(
                [f1_data["x1_orig"].reshape(-1, 1), f1_data["x2_orig"].reshape(-1, 1)]
            ),
            pareto_front=np.hstack(
                [f1_data["f1_orig"].reshape(-1, 1), f1_data["f2_orig"].reshape(-1, 1)]
            ),
            normalized_decision_space=(f1_data["norm_x1"], f1_data["norm_x2"]),
            normalized_objective_space=(f1_data["norm_f1"], f1_data["norm_f2"]),
            parallel_coordinates_data=np.hstack(
                [
                    f1_data["norm_x1"].reshape(-1, 1),
                    f1_data["norm_x2"].reshape(-1, 1),
                    f1_data["norm_f1"].reshape(-1, 1),
                    f1_data["norm_f2"].reshape(-1, 1),
                ]
            ),
            f1_relationships={
                "f1_vs_f2": f1_data["interpolations"]["f1_vs_f2"],
                "f1_vs_x1": f1_data["interpolations"]["f1_vs_x1"],
                "f1_vs_x2": f1_data["interpolations"]["f1_vs_x2"],
                "norm_f1": f1_data["norm_f1"],
            },
            f2_relationships={
                "f2_vs_x1": interp_data["interpolations"]["f2_vs_x1"],
                "f2_vs_x2": interp_data["interpolations"]["f2_vs_x2"],
                "norm_f2": interp_data["norm_f2"],
            },
            x1_x2_relationship={
                "x1": interp_data["norm_x1"],
                "x2": interp_data["norm_x2"],
                "interpolations": interp_data["interpolations"]["x1_vs_x2"],
            },
            multivariate_interpolations=interp_data["multivariate_interpolations"],
        )

    @staticmethod
    def _validate_data(f1_data: dict, interp_data: dict):
        """Validate required keys exist in the source data"""
        required_f1_keys = {
            "f1_orig",
            "f2_orig",
            "x1_orig",
            "x2_orig",
            "norm_f1",
            "norm_f2",
            "norm_x1",
            "norm_x2",
            "interpolations",
        }
        required_interp_keys = {
            "x1_orig",
            "x2_orig",
            "f2_orig",
            "norm_x1",
            "norm_x2",
            "norm_f2",
            "interpolations",
            "multivariate_interpolations",
        }

        if missing := required_f1_keys - f1_data.keys():
            raise ValueError(f"Missing keys in f1_data: {missing}")
        if missing := required_interp_keys - interp_data.keys():
            raise ValueError(f"Missing keys in interp_data: {missing}")
