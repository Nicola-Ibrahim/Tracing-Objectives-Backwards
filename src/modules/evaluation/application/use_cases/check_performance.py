from pydantic import BaseModel, Field

from ....dataset.domain.entities.dataset import Dataset
from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....modeling.domain.entities.trained_pipeline import TrainedPipeline
from ....modeling.domain.interfaces.base_repository import BaseTrainedPipelineRepository
from ....modeling.domain.services.preprocessing_service import PreprocessingService
from ...domain.interfaces.base_visualizer import BaseVisualizer
from .models import InverseEstimatorCandidate


class CheckModelPerformanceParams(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier associated with the model.",
        examples=["dataset"],
    )
    estimator: InverseEstimatorCandidate = Field(
        ...,
        description="Estimator type and optional version number (e.g., 1). If None, latest is used.",
    )
    n_samples: int = Field(
        default=2,
        ge=1,
        description="Number of samples to generate for visualization.",
        examples=[50],
    )

    class Config:
        use_enum_values = True


class TransformChain:
    """Wrapper that composes multiple transforms into a single normalizer interface."""

    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, X):
        for t in self.transforms:
            X = t.transform(X)
        return X

    def inverse_transform(self, X):
        for t in reversed(self.transforms):
            X = t.inverse_transform(X)
        return X


class CheckModelPerformanceService:
    def __init__(
        self,
        model_repository: BaseTrainedPipelineRepository,
        data_repository: BaseDatasetRepository,
        visualizer: BaseVisualizer,
        preprocessing_service: PreprocessingService,
    ):
        self._model_repository = model_repository
        self._data_repository = data_repository
        self._visualizer = visualizer
        self._preprocessing_service = preprocessing_service

    def execute(self, params: CheckModelPerformanceParams) -> None:
        pipeline: TrainedPipeline = self._model_repository.get_version_by_number(
            estimator_type=params.estimator.type.value,
            mapping_direction="inverse",
            dataset_name=params.dataset_name,
            version=params.estimator.version,
        )

        # Load dataset
        dataset: Dataset = self._data_repository.load(name=params.dataset_name)

        X_raw = dataset.objectives
        y_raw = dataset.decisions

        # Split using the pipeline's stored split
        # We explicitly do not apply transforms yet because the visualizer needs raw data
        split_step, X_train, X_test, y_train, y_test = (
            self._preprocessing_service.split(X_raw, y_raw, pipeline.split)
        )

        obj_transforms = pipeline.get_objectives_transforms()
        dec_transforms = pipeline.get_decisions_transforms()

        # 2) Visualize the model performance and fitted curve
        payload = {
            "estimator": pipeline.model.fitted,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_normalizer": TransformChain(obj_transforms),
            "y_normalizer": TransformChain(dec_transforms),
            "non_linear": False,  # or True to try UMAP if installed
            "n_samples": params.n_samples,
            "title": f"Fitted {pipeline.model.fitted.type} (inverse mapping)",
            "training_history": pipeline.model.training_log.model_dump()
            if pipeline.model.training_log
            else {},
            "dataset_name": dataset.name,
        }

        self._visualizer.plot(data=payload)
