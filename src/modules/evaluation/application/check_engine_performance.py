from pydantic import BaseModel, Field

from ...dataset.domain.entities.dataset import Dataset
from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...inverse.domain.entities.inverse_mapping_engine import (
    InverseMappingEngine,
)
from ...inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ...shared.result import Result
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
        default=10,
        ge=1,
        description="Number of samples to generate for visualization.",
    )


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

    def execute(self, params: CheckModelPerformanceParams) -> Result[dict]:
        try:
            engine: InverseMappingEngine = self._engine_repository.load(
                dataset_name=params.dataset_name,
                solver_type=params.engine.solver_type,
                version=params.engine.version,
            )

            # Load dataset
            dataset: Dataset = self._data_repository.load(name=params.dataset_name)

            X_train, y_train = dataset.get_train_data()
            X_test, y_test = dataset.get_test_data()

            # 2) Visualize the model performance and fitted curve
            payload = {
                "estimator": engine.solver,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "X_transform": [
                    engine.transform_decision,
                    engine.inverse_transform_decision,
                ],
                "y_transform": [
                    engine.transform_objective,
                    engine.inverse_transform_objective,
                ],
                "non_linear": False,
                "n_samples": params.n_samples,
                "title": f"Fitted {engine.solver.type()} (v{params.engine.version or 'latest'})",
                "training_history": {},
                "dataset_name": dataset.name,
            }

            self._visualizer.plot(data=payload)

            return Result.ok(
                {
                    "dataset_name": dataset.name,
                    "solver_type": engine.solver.type(),
                    "version": params.engine.version or 0,
                    "insights": {
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                    },
                }
            )
        except FileNotFoundError as e:
            return Result.fail(
                message=f"Dataset or engine not found for {params.dataset_name}",
                details=str(e),
                code="NOT_FOUND",
            )
        except Exception as e:
            return Result.fail(
                message=f"Performance check failed: {str(e)}",
                details=str(e),
                code="INTERNAL_ERROR",
            )
