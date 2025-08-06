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

        with np.load(load_path, allow_pickle=True) as data:
            # Use .get() to handle optional historical data and ensure backward compatibility
            pareto_data = ParetoDataModel(
                pareto_set=data["pareto_set"],
                pareto_front=data["pareto_front"],
                historical_solutions=data.get("historical_solutions"),
                historical_objectives=data.get("historical_objectives"),
                problem_name=str(data["problem_name"]),
                metadata=dict(data["metadata"].item()),
            )
        return pareto_data
