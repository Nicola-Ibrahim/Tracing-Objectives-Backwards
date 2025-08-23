from ...application.analysis.visualize_models_performace.visualize_models_performance_command import (
    VisualizeModelsPerformanceCommand,
)
from ...application.analysis.visualize_models_performace.visualize_models_performance_handler import (
    VisualizeModelsPerformanceCommandHandler,
)
from ...domain.analysis.loaders.interpolator_metrics import InterpolatorMetricsLoader
from ...infrastructure.visualizers.model_artifcat_metrics import (
    PlotlyModelArtifactMetricsVisualizer,
)


def main():
    """
    Main function to run the application.
    It composes the components and executes the workflow.
    """
    # --- Composition Root / Dependency Injection ---

    # 2. Instantiate Infrastructure Layer components
    metrics_loader = InterpolatorMetricsLoader()
    performance_visualizer = PlotlyModelArtifactMetricsVisualizer()

    # 3. Instantiate Application Layer components, injecting dependencies
    performance_handler = VisualizeModelsPerformanceCommandHandler(
        metrics_loader=metrics_loader, visualizer=performance_visualizer
    )

    # 4. Create the command to be handled, now using Pydantic's instantiation
    command = VisualizeModelsPerformanceCommand(dir_name="models")

    # Execute the command by calling the handler.
    performance_handler.handle(command)


if __name__ == "__main__":
    main()
