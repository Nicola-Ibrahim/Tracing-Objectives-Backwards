import numpy as np
from scipy.spatial import geometric_slerp

from ...domain.entities.objectives_preference import ObjectivePreferences
from ...domain.interfaces.base_interpolator import BaseInterpolator
from ..similarities import SimilarityMethod


class GeodesicInterpolator(BaseInterpolator):
    def __init__(self, pareto_set):
        self.pareto_set = pareto_set
        self.start_point = pareto_set[0]
        self.end_point = pareto_set[-1]
        self.t_values = np.linspace(0, 1, len(pareto_set))

        self.inter = None

    def fit(self) -> None:
        """
        Fit the interpolator to the Pareto set by calculating geodesic points
        between the start and end points.
        """
        if self.inter is not None:
            return
        self.inter = geometric_slerp(self.start_point, self.end_point, t_values)

    def interpolate(self, x):
        return self.inter()
