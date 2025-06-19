from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .....shared.config import ROOT_PATH


class BaseParetoVisualizer(ABC):
    def __init__(self, save_path: Path | None = None):
        self.save_path = save_path or ROOT_PATH / "reports/figures"

    @abstractmethod
    def plot(self, pareto_set: np.ndarray, pareto_front: np.ndarray) -> None:
        """Visualize the Pareto set and front."""
        pass
