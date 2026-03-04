import numpy as np
from pydantic import BaseModel, Field

from ...dataset.domain.entities.dataset import Dataset
from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...inverse.domain.entities.inverse_mapping_engine import (
    InverseMappingEngine,
)
from ...inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ..domain.interfaces.base_visualizer import BaseVisualizer


class InverseEngineCandidate(BaseModel):
    """Represents a specific inverse engine candidate (solver_type and optional version)."""

    solver_type: str = Field(..., examples=["GBPI"])
    version: int | None = Field(
        default=None,
        description="Specific integer version number. If None, latest is used.",
    )


class CheckModelPerformanceParams(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier associated with the engine.",
    )
    engine: InverseEngineCandidate = Field(
        ...,
        description="Engine candidate to check.",
    )
    n_samples: int = Field(
        default=2,
        ge=1,
        description="Number of samples to generate for visualization.",
    )


class TransformWrapper:
    """Adapts TransformPipeline to the visualizer's expected interface."""

    def __init__(self, transform_fn, detransform_fn):
        self._transform_fn = transform_fn
        self._detransform_fn = detransform_fn

    def transform(self, X):
        return self._transform_fn(X)

    def inverse_transform(self, X):
        return self._detransform_fn(X)


class CheckModelPerformanceService:
    def __init__(
        self,
        engine_repository: BaseInverseMappingEngineRepository,
        data_repository: BaseDatasetRepository,
        visualizer: BaseVisualizer,
    ):
        self._engine_repository = engine_repository
        self._data_repository = data_repository
        self._visualizer = visualizer

    def execute(self, params: CheckModelPerformanceParams) -> None:
        engine: InverseMappingEngine = self._engine_repository.load(
            dataset_name=params.dataset_name,
            solver_type=params.engine.solver_type,
            version=params.engine.version,
        )

        # Load dataset
        dataset: Dataset = self._data_repository.load(name=params.dataset_name)

        X_raw = dataset.objectives
        y_raw = dataset.decisions

        # Split using the engine's stored split
        train_idx = engine.data_split.train_indices
        test_idx = engine.data_split.test_indices

        X_train, X_test = (
            X_raw[train_idx],
            X_raw[test_idx] if len(test_idx) > 0 else np.array([]),
        )
        y_train, y_test = (
            y_raw[train_idx],
            y_raw[test_idx] if len(test_idx) > 0 else np.array([]),
        )

        # 2) Visualize the model performance and fitted curve
        payload = {
            "estimator": engine.solver,  # Visualizer might need to call .sample()
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_normalizer": TransformWrapper(
                engine.transform_objective, engine.detransform_objective
            ),
            "y_normalizer": TransformWrapper(
                engine.transform_decision, engine.detransform_decision
            ),
            "non_linear": False,
            "n_samples": params.n_samples,
            "title": f"Fitted {engine.solver.type()} (v{params.engine.version or 'latest'})",
            "training_history": {},  # Solver might not have history now
            "dataset_name": dataset.name,
        }

        self._visualizer.plot(data=payload)
