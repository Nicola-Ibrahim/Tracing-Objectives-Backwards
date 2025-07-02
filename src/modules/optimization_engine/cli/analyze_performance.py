from pathlib import Path

from ..application.analyzing.analyze_interpolators_performace.analyze_interpolators_performance_command import (
    AnalyzeIntrepolatorsPerformanceCommand,
)
from ..application.analyzing.analyze_interpolators_performace.analyze_interpolators_performance_handler import (
    AnalyzeModelPerformanceCommandHandler,
)
from ..domain.services.interpolators_metrics import InterpolatorsMetricsLoader
from ..infrastructure.visualizers.interpolators_metrics import (
    PlotlyIntrepolatorsMetricsVisualizer,
)


def main():
    """
    Main function to run the application.
    It composes the components and executes the workflow.
    """
    # --- Composition Root / Dependency Injection ---

    # 2. Instantiate Infrastructure Layer components
    metrics_loader = InterpolatorsMetricsLoader()
    performance_visualizer = PlotlyIntrepolatorsMetricsVisualizer()

    # 3. Instantiate Application Layer components, injecting dependencies
    performance_handler = AnalyzeModelPerformanceCommandHandler(
        metrics_loader=metrics_loader, visualizer=performance_visualizer
    )

    # 4. Create the command to be handled, now using Pydantic's instantiation
    command = AnalyzeIntrepolatorsPerformanceCommand(dir_name="models")

    # Execute the command by calling the handler.
    performance_handler.handle(command)


if __name__ == "__main__":
    main()
