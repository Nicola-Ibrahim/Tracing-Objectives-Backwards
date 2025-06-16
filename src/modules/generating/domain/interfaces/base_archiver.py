from abc import ABC, abstractmethod
from pathlib import Path

from ....shared.config import ROOT_PATH
from ..entities.pareto_data import ParetoDataModel


class BaseParetoArchiver(ABC):
    def __init__(self, file_path: str | Path = "data/raw"):
        """
        Initialize Pareto data manager.

        Args:
            file_path: Base directory for data operations
            default_normalization_fn: Default normalization function
        """
        self.base_path = ROOT_PATH / file_path

    @abstractmethod
    def save(self, data: ParetoDataModel, filename: str) -> Path: ...

    @abstractmethod
    def load(self, filename: str) -> ParetoDataModel: ...
