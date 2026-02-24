from pydantic import BaseModel, Field

from .....modeling.domain.enums.estimator_type import EstimatorTypeEnum


class InverseEstimatorCandidate(BaseModel):
    """Represents a specific inverse estimator candidate (type and optional version)."""

    type: EstimatorTypeEnum = Field(
        ...,
        examples=[EstimatorTypeEnum.MDN.value],
    )
    version: int = Field(
        default=1,
        description="Specific integer version number (e.g., 1). If None, latest is used.",
        examples=[1],
    )


class CheckModelPerformanceParams(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier associated with the model.",
        examples=["dataset"],
    )
    estimator: InverseEstimatorCandidate = Field(
        ...,
        description="Estimator type and optional version number (e.g., 1). If None, latest is used.",
        examples=[InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=1)],
    )
    n_samples: int = Field(
        default=2,
        ge=1,
        description="Number of samples to generate for visualization.",
        examples=[50],
    )

    class Config:
        use_enum_values = True


from .....dataset.domain.entities.dataset import Dataset
from .....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from .....modeling.domain.entities.model_artifact import ModelArtifact
from .....modeling.domain.interfaces.base_repository import BaseModelArtifactRepository
from ....domain.interfaces.base_visualizer import BaseVisualizer


class CheckModelPerformanceService:
    def __init__(
        self,
        model_artificat_repo: BaseModelArtifactRepository,
        processed_dataset_repo: BaseDatasetRepository,
        visualizer: BaseVisualizer,
    ):
        self._model_artificat_repo = model_artificat_repo
        self._processed_repo = processed_dataset_repo
        self._visualizer = visualizer

    def execute(self, params: CheckModelPerformanceParams) -> None:
        model_artificat: ModelArtifact = (
            self._model_artificat_repo.get_version_by_number(
                estimator_type=params.estimator.type.value,
                mapping_direction="inverse",
                dataset_name=params.dataset_name,
                version=params.estimator.version,
            )
        )

        # Load dataset
        dataset: Dataset = self._processed_repo.load(name=params.dataset_name)
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for visualization."
            )

        # 2) Visualize the model performance and fitted curve
        payload = {
            "estimator": model_artificat.estimator,
            "X_train": dataset.processed.objectives_train,
            "y_train": dataset.processed.decisions_train,
            "X_test": dataset.processed.objectives_test,
            "y_test": dataset.processed.decisions_test,
            "X_normalizer": dataset.processed.objectives_normalizer,
            "y_normalizer": dataset.processed.decisions_normalizer,
            "non_linear": False,  # or True to try UMAP if installed
            "n_samples": params.n_samples,
            "title": f"Fitted {model_artificat.estimator.type} (inverse mapping)",
            "training_history": model_artificat.training_history,
            "dataset_name": dataset.name,
        }

        self._visualizer.plot(data=payload)
