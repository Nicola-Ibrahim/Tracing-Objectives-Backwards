from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..generation.interfaces.base_repository import BaseParetoDataRepository
from ..interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)
from ..interpolation.interfaces.base_repository import (
    BaseInverseDecisionMapperRepository,
)


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

    # These dictionaries now define the relationships and the expected filename suffix for loading.
    # The 'class' is no longer directly instantiated here.
    INTERPOLATION_METHODS_1D: dict[str, Any] = {
        "Pchip": {"filename_suffix": "PchipInverseDecisionMapper.pkl"},
        "Cubic Spline": {"filename_suffix": "CubicSplineInverseDecisionMapper.pkl"},
        "Linear": {"filename_suffix": "LinearInverseDecisionMapper.pkl"},
        "Quadratic": {"filename_suffix": "QuadraticInverseDecisionMapper.pkl"},
        "RBF": {"filename_suffix": "RBFInverseDecisionMapper.pkl"},
    }

    INTERPOLATION_METHODS_ND: dict[str, Any] = {
        "Nearest Neighbor": {"filename_suffix": "NearestNDInverseDecisionMapper.pkl"},
        "Linear ND": {"filename_suffix": "LinearNDInverseDecisionMapper.pkl"},
    }

    _NUM_INTERPOLATION_POINTS_1D = 100
    _NUM_INTERPOLATION_POINTS_2D_GRID = 50

    def __init__(
        self,
        pareto_data_repo: BaseParetoDataRepository,
        inverse_decisoin_mapper_repo: BaseInverseDecisionMapperRepository,
    ):
        self._pareto_data_repo = pareto_data_repo
        self._inverse_decisoin_mapper_repo = inverse_decisoin_mapper_repo

    def prepare_dataset(self, data_identifier: str | Path) -> ParetoDataset:
        """Prepare complete Pareto dataset with original, normalized, and interpolated data"""
        loaded_result = self._pareto_data_repo.load(data_identifier)

        if not hasattr(loaded_result, "pareto_set") or not hasattr(
            loaded_result, "pareto_front"
        ):
            raise ValueError(
                "Archiver did not return valid Pareto data (missing pareto_set or pareto_front)."
            )

        if loaded_result.pareto_set.size == 0 or loaded_result.pareto_front.size == 0:
            raise ValueError(
                "Loaded Pareto data (pareto_set or pareto_front) is empty."
            )

        dataset = ParetoDataset()
        dataset.pareto_set = loaded_result.pareto_set
        dataset.pareto_front = loaded_result.pareto_front

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

    def _compute_all_1d_interpolations(self, dataset: ParetoDataset):
        """Load 1D inverse decision mappers and predict their interpolated data for various relationships."""
        norm = dataset.normalized

        # Define interpolation ranges based on the normalized data's actual bounds
        f1_min, f1_max = norm["f1"].min(), norm["f1"].max()
        x1_min, x1_max = norm["x1"].min(), norm["x1"].max()
        f2_min, f2_max = norm["f2"].min(), norm["f2"].max()

        interpolation_ranges_1d = {
            "f1_vs_f2": self._create_interpolation_range(f1_min, f1_max),
            "f1_vs_x1": self._create_interpolation_range(f1_min, f1_max),
            "f1_vs_x2": self._create_interpolation_range(f1_min, f1_max),
            "x1_vs_x2": self._create_interpolation_range(x1_min, x1_max),
            "f2_vs_x1": self._create_interpolation_range(f2_min, f2_max),
            "f2_vs_x2": self._create_interpolation_range(f2_min, f2_max),
        }

        # Explicitly define which relationships support which 1D methods
        relationships_to_process_1d = {
            "f1_vs_f2": ["Pchip", "Cubic Spline", "Linear", "Quadratic", "RBF"],
            "f1_vs_x1": ["Pchip", "Cubic Spline", "Linear", "Quadratic", "RBF"],
            "f1_vs_x2": ["Pchip", "Cubic Spline", "Linear", "Quadratic", "RBF"],
            "x1_vs_x2": ["Pchip", "Cubic Spline", "Linear", "Quadratic", "RBF"],
            "f2_vs_x1": ["Pchip", "Cubic Spline", "Linear", "Quadratic", "RBF"],
            "f2_vs_x2": ["Pchip", "Cubic Spline", "Linear", "Quadratic", "RBF"],
        }

        for relationship_name, methods in relationships_to_process_1d.items():
            # Initialize the dictionary for this relationship if it doesn't exist
            if relationship_name not in dataset.interpolations_1d:
                dataset.interpolations_1d[relationship_name] = {}

            x_interpolation_range = interpolation_ranges_1d[relationship_name]

            for method_name in methods:
                method_info = self.INTERPOLATION_METHODS_1D.get(method_name)
                if not method_info:
                    print(
                        f"Warning: Configuration for 1D method '{method_name}' not found. Skipping."
                    )
                    continue

                result = self._load_and_predict_1d_mapper(
                    relationship_name=relationship_name,
                    method_name=method_name,
                    method_info=method_info,
                    x_interpolation_range=x_interpolation_range,
                )
                if result is not None:
                    dataset.interpolations_1d[relationship_name][method_name] = result

    def _create_interpolation_range(self, min_val: float, max_val: float) -> np.ndarray:
        """Helper to create a linspace for interpolation."""
        if min_val == max_val:
            return np.array([min_val])
        return np.linspace(min_val, max_val, self._NUM_INTERPOLATION_POINTS_1D)

    def _load_and_predict_1d_mapper(
        self,
        relationship_name: str,
        method_name: str,
        method_info: dict[str, Any],
        x_interpolation_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Loads a 1D inverse decision mapper, performs prediction, and returns results as tuples.
        Returns None if the model cannot be loaded or predicted.
        """
        # Filenames follow a convention: relationship_name_MapperClassName.pkl
        # E.g., "f1_vs_f2_PchipInverseDecisionMapper.pkl"
        filename = f"{relationship_name}_{method_info['filename_suffix']}"

        try:
            mapper_instance: BaseInverseDecisionMapper = (
                self._inverse_decisoin_mapper_repo.load(filename)
            )

            if x_interpolation_range.size > 0:
                # Assuming 1D mappers can handle 1D input for prediction.
                # If a mapper (like RBF) expects 2D input for predict even for 1D case,
                # it should be handled within the mapper's predict method or here.
                # For RBF, if it expects (N, 1) and x_interpolation_range is (N,), reshape it.
                if method_name == "RBF":  # Example specific handling
                    x_for_predict = x_interpolation_range.reshape(-1, 1)
                else:
                    x_for_predict = x_interpolation_range

                interpolated_y_values = mapper_instance.predict(x_for_predict)
            else:
                interpolated_y_values = np.array([])

            return (x_interpolation_range, interpolated_y_values)

        except FileNotFoundError:
            print(
                f"Pre-trained model for {relationship_name} - {method_name} not found at {self._inverse_decisoin_mapper_repo.get_model_path(filename)}. Skipping."
            )
            return None
        except Exception as e:
            print(
                f"Error loading or predicting with mapper for {relationship_name} - {method_name}: {str(e)}"
            )
            return None

    def _compute_all_2d_interpolations(self, dataset: ParetoDataset):
        """Load all 2D multivariate inverse decision mappers and predict their interpolated data."""
        X_input_independent_source_data = np.column_stack(
            (dataset.normalized["f1"], dataset.normalized["f2"])
        )

        mesh_f1, mesh_f2, points_for_prediction_2d = (
            np.array([]),
            np.array([]),
            np.array([]),
        )  # Initialize for empty data case

        if X_input_independent_source_data.shape[0] > 0:
            f1_min, f1_max = (
                X_input_independent_source_data[:, 0].min(),
                X_input_independent_source_data[:, 0].max(),
            )
            f2_min, f2_max = (
                X_input_independent_source_data[:, 1].min(),
                X_input_independent_source_data[:, 1].max(),
            )

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

            if grid_f1.size == 1 and grid_f2.size == 1:
                mesh_f1, mesh_f2 = (
                    np.array([grid_f1[0]]),
                    np.array([grid_f2[0]]),
                )  # Ensure they are 1x1 arrays, not scalars
            else:
                mesh_f1, mesh_f2 = np.meshgrid(grid_f1, grid_f2)

            points_for_prediction_2d = np.column_stack(
                (mesh_f1.ravel(), mesh_f2.ravel())
            )

        # Explicitly define which relationships support which 2D methods
        relationships_to_process_2d = {
            "f1f2_vs_x1": ["Nearest Neighbor", "Linear ND"],
            "f1f2_vs_x2": ["Nearest Neighbor", "Linear ND"],
        }

        for relationship_name, methods in relationships_to_process_2d.items():
            # Initialize the dictionary for this relationship if it doesn't exist
            if relationship_name not in dataset.interpolations_2d:
                dataset.interpolations_2d[relationship_name] = {}

            for method_name in methods:
                method_info = self.INTERPOLATION_METHODS_ND.get(method_name)
                if not method_info:
                    print(
                        f"Warning: Configuration for 2D method '{method_name}' not found. Skipping."
                    )
                    continue

                result = self._load_and_predict_2d_mapper(
                    relationship_name=relationship_name,
                    method_name=method_name,
                    method_info=method_info,
                    mesh_f1=mesh_f1,
                    mesh_f2=mesh_f2,
                    points_for_prediction=points_for_prediction_2d,
                )
                if result is not None:
                    dataset.interpolations_2d[relationship_name][method_name] = result

    def _load_and_predict_2d_mapper(
        self,
        relationship_name: str,
        method_name: str,
        method_info: dict[str, Any],
        mesh_f1: np.ndarray,
        mesh_f2: np.ndarray,
        points_for_prediction: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Loads a 2D inverse decision mapper, performs prediction, and returns results as tuples.
        Returns None if the model cannot be loaded or predicted.
        """
        filename = f"{relationship_name}_{method_info['filename_suffix']}"

        try:
            mapper_instance: BaseInverseDecisionMapper = (
                self._inverse_decisoin_mapper_repo.load(filename)
            )

            if points_for_prediction.shape[0] > 0:
                interpolated_z_values_flat = mapper_instance.predict(
                    points_for_prediction
                )
                interpolated_z_values = interpolated_z_values_flat.reshape(
                    mesh_f1.shape
                )
            else:
                interpolated_z_values = np.array([])

            return (mesh_f1, mesh_f2, interpolated_z_values)

        except FileNotFoundError:
            print(
                f"Pre-trained model for {relationship_name} - {method_name} not found at {self._inverse_decisoin_mapper_repo.get_model_path(filename)}. Skipping."
            )
            return None
        except Exception as e:
            print(
                f"Error loading or predicting with mapper for {relationship_name} - {method_name}: {str(e)}"
            )
            return None
