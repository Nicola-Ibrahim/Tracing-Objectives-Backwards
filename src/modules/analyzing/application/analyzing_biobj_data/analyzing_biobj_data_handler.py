from ....shared.adapters.archivers.base import BaseParetoArchiver
from .analyzing_biobj_data_command import AnalyzeBiobjDataCommand


class AnalyzeBiobjDataHandler:
    def __init__(self, archiver: BaseParetoArchiver, plotter):
        self._archiver = archiver
        self._plotter = plotter

    def execute(self, command: AnalyzeBiobjDataCommand):
        results = self._archiver.load(command.results_path)
        self._plotter.plot(results, command.output_path)
