import datetime
import json
from pathlib import Path
from typing import Any

from .base import BaseFileHandler


class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder to serialize datetime objects to ISO 8601 format.
    """

    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
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
                json.dump(obj, f, indent=4, cls=DateTimeEncoder)
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
