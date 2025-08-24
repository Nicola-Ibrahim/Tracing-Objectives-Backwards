from pathlib import Path

import numpy as np

from ....domain.generation.entities.data_model import DataModel
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ...processing.files.npz import NPZFileHandler


class FileSystemDataModelRepository(BaseParetoDataRepository):
    """
    Archiver that saves Pareto data in a compressed numpy format (.npz).
    It now depends on a dedicated NPZFileHandler for file operations.
    """

    def __init__(self):
        super().__init__()
        self._file_handler = NPZFileHandler()

    def save(self, data: DataModel) -> Path:
        """
        Save Pareto set/front data with metadata in numpy format.
        """
        save_path = self.base_path / f"{data.name}.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self._file_handler.save(data.model_dump(), file_path=save_path)
        return save_path

    def load(self, filename: str) -> DataModel:
        """
        Load Pareto set/front data from a numpy archive.
        """
        load_path = self.base_path / f"{filename}.npz"

        loaded_data = self._file_handler.load(load_path)

        # Handle potential None values and coercion from loaded data
        pareto_set = loaded_data.get("pareto_set")
        pareto_front = loaded_data.get("pareto_front")
        historical_solutions = loaded_data.get("historical_solutions")
        historical_objectives = loaded_data.get("historical_objectives")
        name = loaded_data.get("name")
        metadata_raw = loaded_data.get("metadata")

        # Coerce loaded data into expected types
        name = str(name) if name is not None else ""
        metadata = (
            dict(metadata_raw) if isinstance(metadata_raw, (dict, np.ndarray)) else {}
        )

        return DataModel.create(
            name=name,
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            historical_solutions=historical_solutions,
            historical_objectives=historical_objectives,
            metadata=metadata,
        )
