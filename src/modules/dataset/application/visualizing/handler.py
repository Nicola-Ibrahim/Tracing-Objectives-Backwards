from ...domain.entities.dataset import Dataset
from ...domain.interfaces.base_repository import BaseDatasetRepository
from ...domain.interfaces.base_visualizer import BaseVisualizer
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
        dataset_name = command.dataset_name

        dataset: Dataset = self._dataset_repo.load(name=dataset_name)
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for visualization."
            )
        processed = dataset.processed

        # 2) Package payload (arrays only)
        payload = {
            "dataset_name": dataset.name,
            "X_train": processed.decisions_train,
            "y_train": processed.objectives_train,
            "pareto_set": dataset.pareto.set,
            "pareto_front": dataset.pareto.front,
            "historical_solutions": dataset.decisions,
            "historical_objectives": dataset.objectives,
        }

        # 4) Plot
        self._visualizer.plot(data=payload)
