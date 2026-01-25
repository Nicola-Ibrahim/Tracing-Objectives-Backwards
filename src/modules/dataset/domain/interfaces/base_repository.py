from abc import ABC, abstractmethod
from pathlib import Path

from .....shared.config import ROOT_PATH
from ..entities.dataset import Dataset


class BaseDatasetRepository(ABC):
    def __init__(self, file_path: str | Path = "data"):
        """
        Initialize repository with a base directory anchored at the project root.

        Args:
            file_path: Relative directory where dataset bundles are stored.
        """
        self.base_path = ROOT_PATH / file_path

    @abstractmethod
    def save(self, dataset: Dataset) -> Path:
        """Persist the dataset aggregate."""

    @abstractmethod
    def load(self, name: str) -> Dataset:
        """Retrieve the dataset aggregate."""
