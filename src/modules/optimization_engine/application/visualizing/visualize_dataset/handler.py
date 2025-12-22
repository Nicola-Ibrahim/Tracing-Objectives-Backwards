from ....domain.common.interfaces.base_visualizer import BaseVisualizer
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from .command import VisualizeDatasetCommand


class VisualizeDatasetCommandHandler:
    """
    Loads a bi-objective dataset, builds normalizers via an injected factory,
    normalizes decisions (X) and objectives (F), and passes everything to the
    visualizer as a compact dict.

    """

    def __init__(
        self,
        dataset_repo: BaseDatasetRepository,
        visualizer: BaseVisualizer,
    ):
        self._dataset_repo = dataset_repo
        self._visualizer = visualizer

    def execute(self, command: VisualizeDatasetCommand) -> None:
        # 1) Load dataset aggregate
        # We assume command.data_file_name or processed_file_name refers to the dataset name now.
        # Ideally the command should have a single dataset_name.
        # Historically it had two, let's assume one is sufficient or they match.
        dataset_name = command.data_file_name  # or processed_file_name

        dataset: Dataset = self._dataset_repo.load(name=dataset_name)
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for visualization."
            )
        processed = dataset.processed

        # 3) Package payload (arrays only)
        payload = {
            "X_train": processed.decisions_train,
            "y_train": processed.objectives_train,
            "X_test": processed.decisions_test,
            "y_test": processed.objectives_test,
            "pareto_set": dataset.pareto.set,
            "pareto_front": dataset.pareto.front,
            "historical_solutions": dataset.decisions,
            "historical_objectives": dataset.objectives,
        }

        # 7) Plot
        self._visualizer.plot(data=payload)
