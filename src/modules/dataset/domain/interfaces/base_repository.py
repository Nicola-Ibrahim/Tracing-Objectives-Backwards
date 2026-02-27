from abc import ABC, abstractmethod
from pathlib import Path

from ..entities.dataset import Dataset


class BaseDatasetRepository(ABC):
    """
    Abstract repository for persisting and loading Dataset aggregates.

    Concrete implementations are responsible for providing storage location
    and handling serialization details.
    """

    @abstractmethod
    def save(self, dataset: Dataset) -> Path:
        """Persist the dataset aggregate. Returns the path where it was saved."""

    @abstractmethod
    def load(self, name: str) -> Dataset:
        """Retrieve the dataset aggregate by name."""

    @abstractmethod
    def list_all(self) -> list[str]:
        """List all available dataset names."""
