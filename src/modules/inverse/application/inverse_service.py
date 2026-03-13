import time
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...modeling.infrastructure.factories.transformer import TransformerFactory
from ...shared.domain.interfaces.base_logger import BaseLogger
from ...shared.result import Result
from ..domain.entities.inverse_mapping_engine import InverseMappingEngine
from ..domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ..domain.services.generation import CandidateGeneration
from ..domain.value_objects.transform_pipeline import TransformPipeline
from ..infrastructure.solvers.factory import SolversFactory


class SolverConfig(BaseModel):
    """Configuration for the inverse solver."""

    type: str = Field(..., description="Solver type (e.g., GBPI)")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Custom solver parameters"
    )


class TrainInverseMappingEngineParams(BaseModel):
    """Configuration for preparing the generation context."""

    dataset_name: str = Field(..., description="Name of the dataset to load")
    solver: SolverConfig = Field(
        ..., description="Inverse solver strategy configuration"
    )
    transforms: list[dict[str, Any]] = Field(
        default_factory=list, description="Ordered list of transformer configurations"
    )


class GenerationConfig(BaseModel):
    """
    User-configurable parameters for the generation pipeline.
    """

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., description="Name of the dataset")
    target_objective: tuple[float, float] = Field(
        ..., description="Target objective coordinates (2D)"
    )
    solver_type: str = Field(default="GBPI", description="Type of solver to use")
    version: int | None = Field(
        default=None, description="Specific engine version number to use (optional)"
    )
    n_samples: int = Field(
        default=10, ge=1, description="Number of Dirichlet weight samples"
    )


class InverseService:
    """
    Consolidated application service for inverse mapping operations.
    Handles training engines, generating candidates, and listing engines.
    """

    def __init__(
        self,
        dataset_repository: BaseDatasetRepository,
        inverse_mapping_engine_repository: BaseInverseMappingEngineRepository,
        logger: BaseLogger,
        transformer_factory: TransformerFactory,
        solvers_factory: SolversFactory,
    ):
        self._dataset_repository = dataset_repository
        self._inverse_mapping_engine_repository = inverse_mapping_engine_repository
        self._logger = logger
        self._transformer_factory = transformer_factory
        self._solvers_factory = solvers_factory

    def train_engine(self, params: TrainInverseMappingEngineParams) -> Result[dict]:
        """
        Orchestrates training of an inverse mapping engine.
        """
        try:
            start_time = time.time()
            self._logger.log_info(
                f"Preparing generation context for '{params.dataset_name}'"
            )

            dataset = self._dataset_repository.load(params.dataset_name)
            X, y = dataset.get_train_data()

            self._logger.log_info(
                f"Retrieved pre-computed split from dataset: {len(X)} training"
            )

            all_fitted_transforms = []
            for t_cfg in params.transforms:
                target_type = t_cfg.get("target")
                transform = self._transformer_factory.create(t_cfg)
                all_fitted_transforms.append((target_type, transform))

            X_norm = X.copy()
            for label, t in all_fitted_transforms:
                if label == "decisions":
                    t.fit(X_norm)
                    X_norm = t.transform(X_norm)

            y_norm = y.copy()
            for label, t in all_fitted_transforms:
                if label == "objectives":
                    t.fit(y_norm)
                    y_norm = t.transform(y_norm)

            solver = self._solvers_factory.create(
                params.solver.type, params.solver.params
            )
            solver.train(X=X_norm, y=y_norm)

            transform_pipeline = TransformPipeline(transforms=all_fitted_transforms)

            engine = InverseMappingEngine.create(
                dataset_name=params.dataset_name,
                solver=solver,
                transform_pipeline=transform_pipeline,
            )

            version = self._inverse_mapping_engine_repository.save(engine)
            self._logger.log_info(
                f"InverseMappingEngine saved successfully (v{version})."
            )

            duration = time.time() - start_time
            transform_summary = [
                f"{t.__class__.__name__}({label})" for label, t in all_fitted_transforms
            ]

            return Result.ok(
                {
                    "dataset_name": params.dataset_name,
                    "solver_type": params.solver.type,
                    "engine_version": version,
                    "status": "completed",
                    "duration_seconds": duration,
                    "n_train_samples": dataset.metadata.n_train,
                    "n_test_samples": dataset.metadata.n_test,
                    "split_ratio": dataset.metadata.split_ratio,
                    "transform_summary": transform_summary,
                    "training_history": solver.history(),
                }
            )
        except FileNotFoundError as e:
            return Result.fail(
                message=f"Dataset '{params.dataset_name}' not found",
                details=str(e),
                code="NOT_FOUND",
            )
        except Exception as e:
            return Result.fail(
                message="Engine training failed",
                details=str(e),
                code="INTERNAL_ERROR",
            )

    def generate_candidates(self, config: GenerationConfig) -> Result[dict]:
        """
        Generates candidate designs for a target objective using a trained engine.
        """
        try:
            self._logger.log_info(
                f"Starting generation for '{config.dataset_name}' with target {config.target_objective}"
            )

            engine = self._inverse_mapping_engine_repository.load(
                config.dataset_name,
                solver_type=config.solver_type,
                version=config.version,
            )

            result = CandidateGeneration.generate(
                engine=engine,
                target_objective=config.target_objective,
                n_samples=config.n_samples,
            )

            self._logger.log_info(
                f"Generated {result.candidate_objectives.shape[0]} candidates. "
                f"Winner @{result.best_index}: {result.best_candidate_objective.flatten()}"
            )

            return Result.ok(
                {
                    "solver_type": config.solver_type,
                    "target_objective": config.target_objective,
                    "candidate_decisions": result.candidate_decisions.tolist(),
                    "candidate_objectives": [
                        tuple(obj) for obj in result.candidate_objectives.tolist()
                    ],
                    "best_index": result.best_index,
                    "best_candidate_objective": result.best_candidate_objective.flatten().tolist(),
                    "best_candidate_decision": result.best_candidate_decision.flatten().tolist(),
                    "best_candidate_residual": result.best_candidate_residual,
                    "metadata": result.metadata,
                }
            )
        except FileNotFoundError as e:
            return Result.fail(
                message=f"Engine for dataset '{config.dataset_name}' not found",
                details=str(e),
                code="NOT_FOUND",
            )
        except Exception as e:
            return Result.fail(
                message="Candidate generation failed",
                details=str(e),
                code="INTERNAL_ERROR",
            )

    def list_engines(
        self, dataset_name: str | None = None
    ) -> Result[List[Dict[str, Any]]]:
        """Lists trained engines. If dataset_name is None, lists all engines."""
        try:
            if dataset_name:
                engines = self._inverse_mapping_engine_repository.list_all(dataset_name)
            else:
                engines = self._inverse_mapping_engine_repository.list_all_versions()

            return Result.ok(
                [
                    {
                        "dataset_name": e.get("dataset_name"),
                        "solver_type": e["solver_type"],
                        "version": e["version"],
                        "created_at": e["created_at"],
                    }
                    for e in engines
                ]
            )
        except Exception as e:
            return Result.fail(
                message="Failed to list engines",
                details=str(e),
                code="INTERNAL_ERROR",
            )

    def get_available_solvers(self) -> Result[List[Dict[str, Any]]]:
        """Returns metadata for all supported inverse solvers."""
        try:
            return Result.ok(self._solvers_factory.get_solver_schemas())
        except Exception as e:
            return Result.fail(
                message="Failed to get available solvers",
                details=str(e),
                code="INTERNAL_ERROR",
            )

    def delete_engines(self, engine_specs: list[dict]) -> Result[dict[str, Any]]:
        """Deletes multiple engines. engine_specs: list of {'dataset_name', 'solver_type', 'version'}"""
        results = []
        for spec in engine_specs:
            try:
                success = (
                    self._inverse_mapping_engine_repository.delete_specific_engine(
                        spec["dataset_name"], spec["solver_type"], spec["version"]
                    )
                )
                if success:
                    results.append({**spec, "status": "deleted"})
                else:
                    results.append({**spec, "status": "not_found"})
            except Exception as e:
                results.append({**spec, "status": "error", "error": str(e)})

        return Result.ok(
            {
                "total_requested": len(engine_specs),
                "deleted_count": sum(1 for r in results if r["status"] == "deleted"),
                "results": results,
            }
        )

    def get_engine_details(
        self, dataset_name: str, solver_type: str, version: int
    ) -> Result[dict]:
        """
        Retrieves full details for a specific engine version.
        """
        try:
            engine = self._inverse_mapping_engine_repository.load(
                dataset_name, solver_type, version
            )
            dataset = self._dataset_repository.load(dataset_name)

            transform_summary = [
                f"{t.__class__.__name__}({label})"
                for label, t in engine.transform_pipeline.transforms
            ]

            # In this architecture, solver hyperparameters are opaque but can be 
            # extracted if the solver stores them. For now, we'll return what's in history 
            # or an empty dict if not specifically tracked.
            history = engine.solver.history()
            
            return Result.ok(
                {
                    "dataset_name": dataset_name,
                    "solver_type": solver_type,
                    "version": version,
                    "created_at": engine.created_at,
                    "n_train_samples": dataset.metadata.n_train,
                    "n_test_samples": dataset.metadata.n_test,
                    "split_ratio": dataset.metadata.split_ratio,
                    "training_history": history,
                    "transform_summary": transform_summary,
                    "hyperparameters": {}, # Placeholder if we decide to expose fitted params
                }
            )
        except FileNotFoundError as e:
            return Result.fail(
                message=f"Engine or dataset not found",
                details=str(e),
                code="NOT_FOUND",
            )
        except Exception as e:
            return Result.fail(
                message="Failed to retrieve engine details",
                details=str(e),
                code="INTERNAL_ERROR",
            )
