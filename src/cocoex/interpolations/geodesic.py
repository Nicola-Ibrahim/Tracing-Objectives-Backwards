from scipy.spatial import geometric_slerp


class GeodesicInterpolator:
    def __init__(self, pareto_set):
        self.pareto_set = pareto_set
        self.start = pareto_set[0]
        self.end = pareto_set[-1]

    def __call__(self, alpha):
        return geometric_slerp(self.start, self.end, alpha)
