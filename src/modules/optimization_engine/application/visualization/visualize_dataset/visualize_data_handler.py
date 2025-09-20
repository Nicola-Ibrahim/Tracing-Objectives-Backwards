from ....domain.datasets.entities.generated_dataset import GeneratedDataset
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.visualization.interfaces.base_visualizer import BaseVisualizer
from .visualize_data_command import VisualizeDatasetCommand


class VisualizeDatasetCommandHandler:
    """
    Loads a bi-objective dataset, builds normalizers via an injected factory,
    normalizes decisions (X) and objectives (F), and passes everything to the
    visualizer as a compact dict.

    """

    def __init__(
        self,
        dataset_repo: BaseDatasetRepository,
        processed_dataset_repo: BaseDatasetRepository,
        visualizer: BaseVisualizer,
    ):
        self._dataset_repo = dataset_repo
        self._processed_repo = processed_dataset_repo
        self._visualizer = visualizer

    def execute(self, command: VisualizeDatasetCommand) -> None:
        # 1) Load raw data from repository
        raw: GeneratedDataset = self._dataset_repo.load(filename=command.data_file_name)
        processed: ProcessedDataset = self._processed_repo.load(
            filename=command.processed_file_name
        )

        # 3) Package payload (arrays only)
        payload = {
            "X_train": processed.X_train,
            "y_train": processed.y_train,
            "X_test": processed.X_test,
            "y_test": processed.y_test,
            "pareto_set": processed.pareto_set,
            "pareto_front": processed.pareto_front,
            "historical_solutions": raw.historical_solutions,
            "historical_objectives": raw.historical_objectives,
        }

        # 7) Plot
        self._visualizer.plot(data=payload)
