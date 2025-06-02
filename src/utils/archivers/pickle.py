import pickle
from pathlib import Path
from typing import Any

from .base import BaseParetoArchiver
from .models import ParetoDataModel


class ParetoPickleArchiver(BaseParetoArchiver):
    """
    Archiver that saves and loads Pareto data using Python's pickle format.

    This format is suitable for storing complex Python objects including
    arrays and metadata, and is efficient for serialization and deserialization.
    """

    def __init__(self, base_path: str | Path = "data/raw"):
        """
        Initialize the archiver with a base directory for file operations.

        Args:
            base_path (str | Path): The root directory where pickle files will be saved or loaded from.
        """
        super().__init__(base_path)

    def save(self, data: ParetoDataModel, filename: str) -> Path:
        """
        Save ParetoDataModel to a .pkl file using pickle serialization.

        Args:
            data (ParetoDataModel): The structured Pareto data object to save.
            filename (str): The name of the file to save the data to (e.g., 'pareto_result.pkl').

        Returns:
            Path: The path to the saved file.
        """
        self.base_path.mkdir(exist_ok=True, parents=True)
        file_path = self.base_path / filename
        with file_path.open("wb") as f:
            pickle.dump(data.model_dump(), f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path

    def load(self, filename: str) -> ParetoDataModel:
        """
        Load a ParetoDataModel object from a .pkl file.

        Args:
            filename (str): The name of the file to load (e.g., 'pareto_result.pkl').

        Returns:
            ParetoDataModel: The loaded and validated Pareto data object.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            ValueError: If the loaded data cannot be parsed into a ParetoDataModel object.
        """
        file_path = self.base_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"No Pareto data found at {file_path}")
        with file_path.open("rb") as f:
            raw_data: dict[str, Any] = pickle.load(f)
        return ParetoDataModel(**raw_data)
