import numpy as np
from .....domain.visualization.interfaces.base_visualizer import BaseVisualizer
from .vis_2d.visualizer import ModelPerformance2DVisualizer
from .vis_nd.visualizer import ModelPerformanceNDVisualizer


class CompositeVisualizer(BaseVisualizer):
    """
    A composite visualizer that delegates to the appropriate visualizer
    based on the dimensionality of the input data.
    """

    def __init__(self) -> None:
        self._vis_2d = ModelPerformance2DVisualizer()
        self._vis_nd = ModelPerformanceNDVisualizer()

    def plot(self, data: dict) -> None:
        """
        Inspects data['X_train'] to determine dimensionality and delegates.
        """
        X_train = np.asarray(data["X_train"])
        
        # Check input dimension (columns)
        if X_train.ndim == 2 and X_train.shape[1] == 2:
            self._vis_2d.plot(data)
        else:
            self._vis_nd.plot(data)
