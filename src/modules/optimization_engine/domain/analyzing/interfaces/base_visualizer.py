from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .....shared.config import ROOT_PATH


class BaseDataVisualizer(ABC):
    def __init__(self, save_path: Path | None = None):
        self.save_path = save_path or ROOT_PATH / "reports/figures"
        self.save_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def plot(self, data: Any) -> None:
        """Visualize the Pareto set and front."""
        pass
