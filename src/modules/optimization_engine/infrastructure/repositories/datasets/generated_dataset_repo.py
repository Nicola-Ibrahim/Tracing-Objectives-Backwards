from pathlib import Path

import numpy as np

from ....domain.datasets.entities.generated_dataset import GeneratedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ...processing.files.pickle import PickleFileHandler


class FileSystemGeneratedDatasetRepository(BaseDatasetRepository):
    """
    Archiver that saves Pareto data in a compressed numpy format (.npz).
    It now depends on a dedicated NPZFileHandler for file operations.
    """

    def __init__(self):
        super().__init__()
        self._file_handler = PickleFileHandler()

    def save(self, data: GeneratedDataset) -> Path:
        """
        Save Pareto set/front data with metadata in pickle format.
        """
        save_path = self.base_path / data.name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self._file_handler.save(data.model_dump(), file_path=save_path)
        return save_path

    def load(self, filename: str) -> GeneratedDataset:
        """
        Load Pareto set/front data from a numpy archive.
        """
        load_path = self.base_path / filename
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

        return GeneratedDataset.create(
            name=name,
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            historical_solutions=historical_solutions,
            historical_objectives=historical_objectives,
            metadata=metadata,
        )
