from abc import abstractmethod
from pathlib import Path
from typing import Any


class BaseVisualizer:
    """
    Abstract base class for dataset visualizers.

    Concrete implementations are responsible for providing a save path
    and creating any required directories.
    """

    def __init__(self, save_path: Path):
        self.save_path = save_path

    @abstractmethod
    def plot(self, data: Any) -> None:
        """Visualize dataset data."""
        raise NotImplementedError("Subclasses must implement this method.")
