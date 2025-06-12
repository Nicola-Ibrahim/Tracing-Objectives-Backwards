from ....shared.infrastructure.archivers.base import BaseParetoArchiver
from ...domain.interfaces.base_visualizer import BaseParetoVisualizer
from .analyzing_biobj_data_command import AnalyzeBiobjDataCommand


class AnalyzeBiobjDataHandler:
    def __init__(self, archiver: BaseParetoArchiver, visualizer: BaseParetoVisualizer):
        self._archiver = archiver
        self._visualizer = visualizer

    def execute(self, command: AnalyzeBiobjDataCommand):
        results = self._archiver.load(command.results_path)
        self._visualizer.plot(results, command.output_path)
