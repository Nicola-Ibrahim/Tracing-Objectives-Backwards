from ...domain.interfaces.base_data_source import BaseDataSource, RawDataPayload
from .base_optimizer import BaseOptimizer


class OptimizationDataSource(BaseDataSource):
    """
    Adapter: wraps a pymoo-based optimizer to implement the
    framework-agnostic BaseDataSource interface.

    This is the bridge between the COCO/pymoo optimization pipeline
    and the domain's data model. It produces a RawDataPayload that
    includes Pareto data because the optimization solver computes it.
    """

    def __init__(self, optimizer: BaseOptimizer):
        self._optimizer = optimizer

    def generate(self) -> RawDataPayload:
        """
        Runs the optimizer and maps the result to a generic RawDataPayload.
        Pareto data is extracted from the solver output.
        """
        opt_data = self._optimizer.run()

        decisions = (
            opt_data.historical_solutions
            if opt_data.historical_solutions is not None
            else opt_data.pareto_set
        )
        objectives = (
            opt_data.historical_objectives
            if opt_data.historical_objectives is not None
            else opt_data.pareto_front
        )

        return RawDataPayload(
            decisions=decisions,
            objectives=objectives,
            pareto_set=opt_data.pareto_set,
            pareto_front=opt_data.pareto_front,
        )
