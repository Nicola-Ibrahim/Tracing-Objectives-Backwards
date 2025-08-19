from pathlib import Path

import numpy as np

from ....domain.generation.entities.pareto_data import ParetoDataModel
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository


class NPZParetoDataRepository(BaseParetoDataRepository):
    """
    Archiver that saves Pareto data in compressed numpy format (.npz).
    This format is efficient for storing large arrays and supports metadata.
    It provides methods to save and load Pareto sets and fronts with optional normalization.
    """

    def save(self, data: ParetoDataModel, filename: str) -> Path:
        """
        Save Pareto set/front data with metadata in numpy format.
        Now includes optional historical data.
        """

        # Ensure directory exists
        save_path = self.base_path / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data dictionary to be saved
        data_to_save = {
            "pareto_set": data.pareto_set,
            "pareto_front": data.pareto_front,
            "historical_solutions": data.historical_solutions,
            "historical_objectives": data.historical_objectives,
            "problem_name": data.problem_name,
            "metadata": data.metadata if data.metadata else {},
        }

        # Save as compressed numpy archive
        np.savez_compressed(save_path, **data_to_save)

        return save_path

    def load(self, filename: str) -> ParetoDataModel:
        """
        Load Pareto set/front data from numpy archive.
        Now supports loading optional historical data.

        Args:
            filename: Name of the file to load
        Returns:
            ParetoDataModel: Loaded Pareto data object containing set, front, problem name, and metadata
        Raises:
            FileNotFoundError: If the specified file does not exist
        """

        load_path = self.base_path / f"{filename}.npz"

        if not load_path.exists():
            raise FileNotFoundError(f"No Pareto data found at {load_path}")

        def _unpack(value, default=None):
            """Helper to extract python objects from numpy 0-d arrays saved with allow_pickle.

            If value is a zero-dim numpy array (common when saving Python objects),
            return its item(); otherwise return value as-is. If value is None, return default.
            """
            if value is None:
                return default
            # numpy zero-d arrays that hold pickled Python objects
            if isinstance(value, np.ndarray) and value.shape == ():
                try:
                    return value.item()
                except Exception:
                    return default
            return value

        with np.load(load_path, allow_pickle=True) as data:
            pareto_set = data["pareto_set"]
            pareto_front = data["pareto_front"]

            historical_solutions = _unpack(data.get("historical_solutions"), None)
            historical_objectives = _unpack(data.get("historical_objectives"), None)

            raw_problem = _unpack(data.get("problem_name"), "")
            problem_name = str(raw_problem) if raw_problem is not None else ""

            raw_metadata = _unpack(data.get("metadata"), {})
            if raw_metadata is None:
                metadata = {}
            elif isinstance(raw_metadata, dict):
                metadata = raw_metadata
            else:
                # Try to coerce mapping-like objects into dict safely
                try:
                    metadata = dict(raw_metadata)
                except Exception:
                    metadata = {}

            pareto_data = ParetoDataModel(
                pareto_set=pareto_set,
                pareto_front=pareto_front,
                historical_solutions=historical_solutions,
                historical_objectives=historical_objectives,
                problem_name=problem_name,
                metadata=metadata,
            )
        return pareto_data
