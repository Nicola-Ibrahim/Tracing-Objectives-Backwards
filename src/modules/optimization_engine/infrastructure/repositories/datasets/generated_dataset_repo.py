from pathlib import Path

from ....domain.datasets.entities.generated_dataset import GeneratedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.datasets.value_objects.pareto import Pareto
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
        # Backwards compatibility with legacy schema
        pareto = Pareto(
            set=loaded_data.get("pareto").get("set"),
            front=loaded_data.get("pareto").get("front"),
        )

        return GeneratedDataset.create(
            name=loaded_data.get("name", ""),
            X=loaded_data.get("X"),
            y=loaded_data.get("y"),
            pareto=pareto,
        )
