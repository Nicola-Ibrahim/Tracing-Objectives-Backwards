import numpy as np

from ...config import ROOT_PATH
from .base import BaseParetoArchiver
from .models import ParetoDataModel


class ParetoNPzArchiver(BaseParetoArchiver):
    """
    Archiver that saves Pareto data in compressed numpy format (.npz).
    This format is efficient for storing large arrays and supports metadata.
    It provides methods to save and load Pareto sets and fronts with optional normalization.
    """

    def save(self, data: ParetoDataModel, filename: str = "pareto_data.npz") -> None:
        """
        Save Pareto set/front data with metadata in numpy format.

        Args:
            pareto_set: Array of decision vectors (n_solutions, n_dim)
            pareto_front: Array of objective vectors (n_solutions, n_obj)
            problem_name: Identifier for the optimization problem
            filename: Target filename for saving
            metadata: Additional metadata to store
            validate: Verify set/front dimension consistency
        """

        # Ensure directory exists
        save_path = ROOT_PATH / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as compressed numpy archive
        np.savez_compressed(
            save_path,
            pareto_set=data.pareto_set,
            pareto_front=data.pareto_front,
            problem_name=data.problem_name,
            metadata=data.metadata if data.metadata else {},
        )

    def load(self, filename: str = "pareto_data.npz") -> ParetoDataModel:
        """
        Load Pareto set/front data from numpy archive.

        Args:
            filename: Name of the file to load
        Returns:
            ParetoDataModel: Loaded Pareto data object containing set, front, problem name, and metadata
        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        load_path = ROOT_PATH / filename
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
