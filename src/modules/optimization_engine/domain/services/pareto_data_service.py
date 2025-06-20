from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..paretos.interfaces.base_archiver import BaseParetoArchiver


class ParetoDataService:
    """
    Application Service responsible for orchestrating the retrieval of raw Pareto data
    and its subsequent preparation into formats suitable for visualization.
    It communicates with the 'paretos' context via the BaseParetoArchiver interface.
    """

    def __init__(self, archiver: BaseParetoArchiver):
        """
        Initializes the service with a dependency on a Pareto Archiver.

        Args:
            archiver: An implementation of BaseParetoArchiver for loading Pareto data.
        """
        self.archiver = archiver

    def provide_visualization_data(
        self, data_identifier: str | Path
    ) -> tuple[dict, dict]:
        """
        Retrieves raw biobjective Pareto data from the archiver,
        and prepares it into structured dictionaries ready for visualization.

        This method encapsulates both data retrieval and the specific preparation steps
        required for the visualization dashboard.

        Args:
            data_identifier: The identifier (e.g., file path, database ID)
                             to locate the Pareto data.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - The prepared data for f1 relationships (dict).
                - The prepared data for x1-x2 interpolation (dict).

        Raises:
            FileNotFoundError: If the data specified by `data_identifier` is not found
                               (propagated from the Archiver).
            ValueError: If the loaded data does not contain expected Pareto components.
            Exception: Any other exception during data retrieval or preparation.
        """
        print(
            f"ParetoDataService: Requesting raw Pareto data for '{data_identifier}' from archiver."
        )

        # 1. Retrieve Raw Data (via archiver)
        loaded_result = self.archiver.load(data_identifier)

        if not hasattr(loaded_result, "pareto_set") or not hasattr(
            loaded_result, "pareto_front"
        ):
            raise ValueError(
                "Archiver did not return a result with 'pareto_set' and 'pareto_front' attributes."
            )

        pareto_set = loaded_result.pareto_set
        pareto_front = loaded_result.pareto_front

        print(
            "ParetoDataService: Raw Pareto data successfully retrieved. Preparing for visualization..."
        )

        # 2. Orchestrate Data Preparation (using internal helpers)
        f1_rel_data = self._prepare_f1_relationships_data(pareto_set, pareto_front)
        x1_x2_interp_data = self._prepare_x1_x2_interpolation_data(pareto_set)

        print("ParetoDataService: Data prepared for visualization.")

        return f1_rel_data, x1_x2_interp_data

    def _normalize_array(self, data_array: np.ndarray) -> np.ndarray:
        """
        Normalizes a single numpy array (1D or 2D) using MinMaxScaler.
        This is a reusable private helper method within the service.
        """
        scaler = MinMaxScaler()
        # Reshape to 2D if the input is 1D, as MinMaxScaler expects 2D input.
        reshaped_data = (
            data_array.reshape(-1, 1) if data_array.ndim == 1 else data_array
        )
        normalized_data = scaler.fit_transform(reshaped_data)
        # Flatten back to 1D if the original was 1D for consistency with previous outputs.
        return normalized_data.flatten() if data_array.ndim == 1 else normalized_data

    def _prepare_f1_relationships_data(
        self, pareto_set: np.ndarray, pareto_front: np.ndarray
    ) -> dict:
        """
        Prepares and returns data needed for f1 relationships plots.
        It extracts specific objective and design variable data, normalizes it,
        and identifies unique points for spline interpolation.
        """
        # Extract raw data points
        f1_orig = pareto_front[:, 0]
        f2_orig = pareto_front[:, 1]
        x1_orig = pareto_set[:, 0]
        x2_orig = pareto_set[:, 1]

        # Normalize extracted data using the helper method
        norm_f1_all = self._normalize_array(f1_orig)
        norm_f2_all = self._normalize_array(f2_orig)
        norm_x1_all = self._normalize_array(x1_orig)
        norm_x2_all = self._normalize_array(x2_orig)

        # Identify unique points based on normalized f1 for spline interpolation.
        # This logic is specific to f1 relationship plots, so it remains here.
        unique_norm_f1, unique_norm_f1_idx = np.unique(norm_f1_all, return_index=True)

        norm_f1_unique_for_spline = norm_f1_all[unique_norm_f1_idx]
        norm_f2_for_spline = norm_f2_all[unique_norm_f1_idx]
        norm_x1_for_spline = norm_x1_all[unique_norm_f1_idx]
        norm_x2_for_spline = norm_x2_all[unique_norm_f1_idx]

        return {
            "f1_orig": f1_orig,
            "f2_orig": f2_orig,
            "x1_orig": x1_orig,
            "x2_orig": x2_orig,
            "norm_f1_all": norm_f1_all,
            "norm_f2_all": norm_f2_all,
            "norm_x1_all": norm_x1_all,
            "norm_x2_all": norm_x2_all,
            "norm_f1_unique_for_spline": norm_f1_unique_for_spline,
            "norm_f2_for_spline": norm_f2_for_spline,
            "norm_x1_for_spline": norm_x1_for_spline,
            "norm_x2_for_spline": norm_x2_for_spline,
        }

    def _prepare_x1_x2_interpolation_data(self, pareto_set: np.ndarray) -> dict:
        """
        Prepares and returns data needed for x1 vs x2 interpolation.
        It extracts design variable data, normalizes it, and identifies
        unique points for spline interpolation.
        """
        # Extract raw data points
        x1_orig = pareto_set[:, 0]
        x2_orig = pareto_set[:, 1]

        # Normalize extracted data using the helper method
        norm_x1_all = self._normalize_array(x1_orig)
        norm_x2_all = self._normalize_array(x2_orig)

        # Identify unique points based on normalized x1 for spline interpolation.
        # This logic is specific to x1-x2 interpolation, so it remains here.
        unique_norm_x1, unique_norm_indices = np.unique(norm_x1_all, return_index=True)
        norm_x2_for_unique_x1 = norm_x2_all[unique_norm_indices]

        return {
            "x1_orig": x1_orig,
            "x2_orig": x2_orig,
            "norm_x1_all": norm_x1_all,
            "norm_x2_all": norm_x2_all,
            "norm_x1_unique_for_spline": unique_norm_x1,
            "norm_x2_for_spline": norm_x2_for_unique_x1,
        }
