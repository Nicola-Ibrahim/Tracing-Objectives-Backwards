from abc import ABC, abstractmethod

import numpy as np


class BaseParetoVisualizer(ABC):
    @abstractmethod
    def plot(self, pareto_set: np.ndarray, pareto_front: np.ndarray) -> None:
        """Visualize the Pareto set and front."""
        pass
