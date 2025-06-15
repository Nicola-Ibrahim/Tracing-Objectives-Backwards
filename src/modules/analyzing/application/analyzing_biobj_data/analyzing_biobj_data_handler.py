from ....shared.infrastructure.archivers.base import BaseParetoArchiver
from ...domain.interfaces.base_visualizer import BaseParetoVisualizer
from .analyzing_biobj_data_command import AnalyzeBiobjDataCommand


class AnalyzeBiobjDataHandler:
    def __init__(self, archiver: BaseParetoArchiver, visualizer: BaseParetoVisualizer):
        self._archiver = archiver
        self._visualizer = visualizer

    def execute(self, command: AnalyzeBiobjDataCommand) -> None:
        result = self._archiver.load(filename=command.filename)
        self._visualizer.plot(
            pareto_set=result.pareto_set, pareto_front=result.pareto_front
        )
