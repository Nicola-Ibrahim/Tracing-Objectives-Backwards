from abc import ABC, abstractmethod
from pathlib import Path

from .....shared.config import ROOT_PATH
from ..entities.data_model import DataModel


class BaseParetoDataRepository(ABC):
    def __init__(self, file_path: str | Path = "data/raw"):
        """
        Initialize Pareto data manager.

        Args:
            file_path: Base directory for data operations
            default_normalization_fn: Default normalization function
        """
        self.base_path = ROOT_PATH / file_path

    @abstractmethod
    def save(self, data: DataModel) -> Path: ...

    @abstractmethod
    def load(self, filename: str) -> DataModel: ...
