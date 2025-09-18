from ...application.visualization.visualize_model_performance.visualize_model_performance_command import (
    VisualizeModelPerformanceCommand,
)
from ...application.visualization.visualize_model_performance.visualize_model_performance_handler import (
    VisualizeModelPerformanceCommandHandler,
)
from ...infrastructure.repositories.datasets.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)
from ...infrastructure.repositories.modeling.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.visualizers.curves import (
    ModelCurveVisualizer,
)

if __name__ == "__main__":
    handler = VisualizeModelPerformanceCommandHandler(
        model_artificat_repo=FileSystemModelArtifactRepository(),
        processed_dataset_repo=FileSystemProcessedDatasetRepository(),
        visualizer=ModelCurveVisualizer(),
    )

    command = VisualizeModelPerformanceCommand(estimator_type="CVAE")

    handler.execute(command)
