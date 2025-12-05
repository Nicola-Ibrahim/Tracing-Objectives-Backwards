from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.entities.processed_data import ProcessedData
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.entities.model_artifact import ModelArtifact
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....domain.visualization.interfaces.base_visualizer import BaseVisualizer
from .command import VisualizeModelPerformanceCommand


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
        if command.model_number:
            all_versions = self._model_artificat_repo.get_all_versions(
                estimator_type=command.estimator_type,
                mapping_direction=command.mapping_direction,
            )
            if not all_versions:
                raise FileNotFoundError(
                    f"No model versions found for type: '{command.estimator_type}' "
                    f"and mapping_direction: '{command.mapping_direction}'"
                )
            index = command.model_number - 1
            if index >= len(all_versions):
                raise IndexError(
                    f"Requested model number {command.model_number} exceeds "
                    f"available versions ({len(all_versions)})."
                )
            model_artificat: ModelArtifact = all_versions[index]
        else:
            model_artificat = self._model_artificat_repo.get_latest_version(
                estimator_type=command.estimator_type,
                mapping_direction=command.mapping_direction,
            )

        # Load dataset
        dataset: Dataset = self._processed_repo.load(name=command.processed_file_name)
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for visualization."
            )
        processed = dataset.processed

        if command.mapping_direction == "inverse":
            X_train = processed.objectives_train
            y_train = processed.decisions_train
            X_test = processed.objectives_test
            y_test = processed.decisions_test
            X_normalizer = processed.objectives_normalizer
            y_normalizer = processed.decisions_normalizer
            mapping_label = "inverse"
        else:
            # forward
            X_train = processed.decisions_train
            y_train = processed.objectives_train
            X_test = processed.decisions_test
            y_test = processed.objectives_test
            X_normalizer = processed.decisions_normalizer
            y_normalizer = processed.objectives_normalizer
            mapping_label = "forward"

        # 2) Visualize the model performance and fitted curve
        payload = {
            "estimator": model_artificat.estimator,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_normalizer": X_normalizer,
            "y_normalizer": y_normalizer,
            "non_linear": False,  # or True to try UMAP if installed
            "n_samples": 300,
            "title": f"Fitted {model_artificat.estimator.type} ({mapping_label} mapping)",
            "loss_history": model_artificat.loss_history,
            "mapping_direction": command.mapping_direction,
        }

        self._visualizer.plot(data=payload)
