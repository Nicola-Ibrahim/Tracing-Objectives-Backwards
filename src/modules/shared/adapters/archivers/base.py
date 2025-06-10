from abc import ABC, abstractmethod
from pathlib import Path

from ....generating.domain.entities.pareto_data import ParetoDataModel
from ...config import ROOT_PATH


class BaseParetoArchiver(ABC):
    def __init__(self, base_path: str | Path = "data/raw"):
        """
        Initialize Pareto data manager.

        Args:
            base_path: Base directory for data operations
            default_normalization_fn: Default normalization function
        """
        self.base_path = ROOT_PATH / base_path

    @abstractmethod
    def save(self, data: ParetoDataModel, filename: str) -> Path: ...

    @abstractmethod
    def load(self, filename: str) -> dict: ...
