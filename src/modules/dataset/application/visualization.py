# --- from visualizing ---
from pydantic import BaseModel, Field

from ..domain.entities.dataset import Dataset
from ..domain.interfaces.base_repository import BaseDatasetRepository
from ..domain.interfaces.base_visualizer import BaseVisualizer


class VisualizeDatasetParams(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier to visualize.",
        examples=["dataset"],
    )


class VisualizeDatasetService:
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

    def execute(self, params: VisualizeDatasetParams) -> None:
        # 1) Load dataset aggregate
        dataset_name = params.dataset_name

        dataset: Dataset = self._dataset_repo.load(name=dataset_name)

        # 2) Package payload (arrays only)
        payload = {
            "dataset_name": dataset.name,
            "X_raw": dataset.decisions,
            "y_raw": dataset.objectives,
            "pareto_set": dataset.pareto.set if dataset.pareto else None,
            "pareto_front": dataset.pareto.front if dataset.pareto else None,
        }

        # 4) Plot
        self._visualizer.plot(data=payload)
