from abc import ABC, abstractmethod
from typing import Any
from zipfile import Path


class BaseFileHandler(ABC):
    @abstractmethod
    def save(self, obj: Any, file_path: Path):
        """Saves a Python object to a specified file path."""
        pass

    @abstractmethod
    def load(self, file_path: Path) -> Any:
        """Loads a Python object from a specified file path."""
        pass
