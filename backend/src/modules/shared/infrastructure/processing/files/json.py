import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseFileHandler


class SciDataEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder to serialize:
    1. datetime objects to ISO 8601 format.
    2. numpy arrays to lists.
    3. numpy scalars to Python primitives.
    """

    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


class JsonFileHandler(BaseFileHandler):
    """
    Handles the serialization and deserialization of model artifacts (using JSON)
    and their associated metadata (using JSON) to and from the file system.
    """

    def save(self, obj: Any, file_path: Path):
        """Saves a Python object to a specified file path using JSON."""
        try:
            with open(file_path, "w") as f:
                json.dump(obj, f, indent=4, cls=SciDataEncoder)
        except Exception as e:
            raise IOError(f"Failed to save object to {file_path}: {e}") from e

    def load(self, file_path: Path) -> Any:
        """Loads a Python object from a specified file path using JSON."""
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found at {file_path}")
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load object from {file_path}: {e}") from e
