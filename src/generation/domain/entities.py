from .value_objects import OptimizationResult


class ParetoFront:
    def __init__(self, result: OptimizationResult):
        self.result = result

    def get_indices(self):
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        return NonDominatedSorting().do(self.result.F)[0]
