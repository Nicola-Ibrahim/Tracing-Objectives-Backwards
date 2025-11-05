from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Union

from .....shared.config import ROOT_PATH
from ..entities.generated_dataset import GeneratedDataset
from ..entities.processed_dataset import ProcessedDataset


class BaseDatasetRepository(ABC):
    def __init__(self, file_path: str | Path = "data"):
        """
        Initialize repository with a base directory anchored at the project root.

        Args:
            file_path: Relative directory where dataset bundles are stored.
        """
        self.base_path = ROOT_PATH / file_path

    @abstractmethod
    def save(
        self, *, raw: GeneratedDataset, processed: ProcessedDataset
    ) -> Path:
        """Persist both raw and processed dataset bundles."""

    @abstractmethod
    def load(
        self, filename: str, variant: Literal["raw", "processed"] = "processed"
    ) -> Union[GeneratedDataset, ProcessedDataset]:
        """Retrieve either the raw or processed dataset variant."""
