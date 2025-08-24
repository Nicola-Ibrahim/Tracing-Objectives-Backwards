import pickle
from pathlib import Path
from typing import Any

from .base import BaseFileHandler


class PickleFileHandler(BaseFileHandler):
    """
    Handles the serialization and deserialization of model artifacts (using pickle)
    and their associated metadata (using JSON) to and from the file system.
    """

    def save(self, obj: Any, file_path: Path):
        """Saves a Python object to a specified file path using pickle."""
        try:
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        except Exception as e:
            raise IOError(f"Failed to pickle object to {file_path}: {e}") from e

    def load(self, file_path: Path) -> Any:
        """Loads a Python object from a specified file path using pickle."""
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found at {file_path}")
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to unpickle object from {file_path}: {e}") from e
