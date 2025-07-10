from ....domain.analysis.interfaces.base_visualizer import (
    BaseDataVisualizer,
)
from ....domain.analysis.loaders.interpolator_metrics import InterpolatorMetricsLoader
from .analyze_interpolators_performance_command import (
    AnalyzeIntrepolatorsPerformanceCommand,
)


class AnalyzeModelPerformanceCommandHandler:
    """
    An orchestrator that handles the end-to-end workflow of fetching metrics,
    grouping them by method, and visualizing their distribution.
    """

    def __init__(
        self,
        metrics_loader: InterpolatorMetricsLoader,
        visualizer: BaseDataVisualizer,
    ):
        """
        Args:
            metrics_loader (InterpolatorMetricsLoader): The data access component (dependency).
            visualizer (BaseDataVisualizer): The plotting component (dependency).
        """
        self.metrics_loader = metrics_loader
        self.visualizer = visualizer

    def handle(self, command: AnalyzeIntrepolatorsPerformanceCommand) -> None:
        """
        Executes the full command workflow.

        Args:
            command (AnalyzeIntrepolatorsPerformanceCommand): The command containing the model directory.
        """

        metrics = self.metrics_loader.fetch(dir_name=command.dir_name)

        self.visualizer.plot(metrics)
