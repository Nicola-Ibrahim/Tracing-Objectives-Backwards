from pathlib import Path

import numpy as np

from ...domain.entities.pareto_data import ParetoDataModel
from ...domain.interfaces.base_archiver import BaseParetoArchiver


class ParetoNPzArchiver(BaseParetoArchiver):
    """
    Archiver that saves Pareto data in compressed numpy format (.npz).
    This format is efficient for storing large arrays and supports metadata.
    It provides methods to save and load Pareto sets and fronts with optional normalization.
    """

    def save(self, data: ParetoDataModel, filename: str = "pareto_data.npz") -> Path:
        """
        Save Pareto set/front data with metadata in numpy format.

        """

        # Ensure directory exists
        save_path = self.base_path / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as compressed numpy archive
        np.savez_compressed(
            save_path,
            pareto_set=data.pareto_set,
            pareto_front=data.pareto_front,
            problem_name=data.problem_name,
            metadata=data.metadata if data.metadata else {},
        )

        return save_path

    def load(self, filename: str = "pareto_data") -> ParetoDataModel:
        """
        Load Pareto set/front data from numpy archive.

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
            pareto_data = ParetoDataModel(
                pareto_set=data["pareto_set"],
                pareto_front=data["pareto_front"],
                problem_name=str(data["problem_name"]),
                metadata=dict(data["metadata"].item()),
            )
        return pareto_data
