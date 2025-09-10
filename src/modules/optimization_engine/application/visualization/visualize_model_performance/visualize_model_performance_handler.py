from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.entities.model_artifact import ModelArtifact
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....domain.visualization.interfaces.base_visualizer import BaseVisualizer
from .visualize_model_performance_command import VisualizeModelPerformanceCommand


class VisualizeModelPerformanceCommandHandler:
    def __init__(
        self,
        model_artificat_repo: BaseModelArtifactRepository,
        processed_dataset_repo: BaseDatasetRepository,
        visualizer: BaseVisualizer,
    ):
        self._model_artificat_repo = model_artificat_repo
        self._processed_repo = processed_dataset_repo
        self._visualizer = visualizer

    def execute(self, command: VisualizeModelPerformanceCommand) -> None:
        # 1) Load raw data and model artifact from repository
        model_artificat: ModelArtifact = self._model_artificat_repo.get_latest_version(
            estimator_type=command.estimator_type
        )
        processed: ProcessedDataset = self._processed_repo.load(
            filename=command.processed_file_name
        )

        # 2) Visualize the model performance and fitted curve
        payload = {
            "estimator": model_artificat.estimator,
            "X_train": processed.X_train,
            "y_train": processed.y_train,
            "X_test": processed.X_test,
            "y_test": processed.y_test,
            "X_normalizer": processed.X_normalizer,
            "y_normalizer": processed.y_normalizer,
            "non_linear": False,  # or True to try UMAP if installed
            "n_samples": 300,
            "title": f"Fitted {model_artificat.estimator.type}",
            "loss_history": model_artificat.loss_history,
        }

        self._visualizer.plot(data=payload)
