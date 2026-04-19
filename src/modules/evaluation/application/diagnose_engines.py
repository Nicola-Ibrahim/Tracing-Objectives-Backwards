import uuid
from typing import Any, AsyncGenerator

import numpy as np

from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...inverse.domain.entities.inverse_mapping_engine import (
    InverseMappingEngine,
)
from ...inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ...shared.domain.interfaces.base_logger import BaseLogger
from ...shared.domain.interfaces.base_task_manager import (
    BaseTaskManager,
)
from ...shared.result import Result
from ..domain.aggregates.diagnostic_report import DiagnosticReport
from ..domain.enums.engine_capability import EngineCapability
from ..domain.enums.mapping_direction import MappingDirection
from ..domain.interfaces.base_diagnostic_repository import (
    BaseDiagnosticRepository,
)
from ..domain.services.decision_space_auditor import DecisionSpaceAuditor
from ..domain.services.objective_space_auditor import ObjectiveSpaceAuditor
from ..domain.value_objects.engine import Engine as EngineVO
from .diagnose_engines_command import (
    EngineCandidate,
    RunDiagnosticsCommand,
)


class RunDiagnosticsService:
    """
    Application service that orchestrates the diagnostic suite.
    Can be run synchronously (execute) or asynchronously (execute_async).
    """

    @staticmethod
    def map_capability(solver_type: str) -> EngineCapability:
        """Determines evaluation capability based on solver type."""
        st = solver_type.upper().strip()
        if st in ["MDN", "CVAE", "INN", "CAVA"]:
            return EngineCapability.FULL_DISTRIBUTION
        if st in ["GBPI"]:
            return EngineCapability.PREDICTION_INTERVAL
        return EngineCapability.FULL_DISTRIBUTION  # Conservative default

    def __init__(
        self,
        engine_repository: BaseInverseMappingEngineRepository,
        data_repository: BaseDatasetRepository,
        diagnostic_repository: BaseDiagnosticRepository,
        task_manager: BaseTaskManager,
        logger: BaseLogger,
    ):
        self._engine_repository = engine_repository
        self._data_repo = data_repository
        self._diag_repo = diagnostic_repository
        self._task_manager = task_manager
        self._logger = logger

    async def execute_async(self, command: RunDiagnosticsCommand) -> str:
        """
        Orchestrates the asynchronous execution:
        1. Generates a task_id.
        2. Initializes status.
        3. Enqueues the worker task.
        """
        task_id = str(uuid.uuid4())

        await self._task_manager.set_status(
            task_id, {"event": "queued", "status": "Task Queued", "progress": 0}
        )

        await self._task_manager.enqueue(
            "run_diagnostics_task",
            task_id=task_id,
            command_data=command.model_dump(),
        )

        return task_id

    async def stream_progress(self, task_id: str) -> AsyncGenerator[str, None]:
        """
        Progress stream delegated to the unified task manager.
        """
        async for chunk in self._task_manager.subscribe(task_id):
            yield chunk

    async def execute(
        self,
        command: RunDiagnosticsCommand,
        task_id: str | None = None,
    ) -> Result[list[DiagnosticReport]]:
        """
        Executes the diagnostic suite.
        Optional task_id allows for async progress tracking via the injected publisher.
        """

        async def publish_safe(data: dict[str, Any]):
            if task_id:
                await self._task_manager.publish(task_id, data)

        try:
            self._logger.log_info(
                f"Starting Diagnostic Compute for engines on '{command.dataset_name}'"
            )

            dataset = self._data_repo.load(command.dataset_name)

            engines_to_diagnose = self._initialize_engines(
                candidates=command.inverse_engine_candidates,
                dataset_name=command.dataset_name,
            )

            reports = []

            for engine, version in engines_to_diagnose:
                engine_type = engine.solver.type()
                engine_name = f"{engine_type} (v{version})"
                self._logger.log_info(f"Diagnosing {engine_name}")

                await publish_safe(
                    {"status": f"Diagnosing {engine_name}", "progress": 0}
                )

                test_indices = dataset.test_indices
                if len(test_indices) == 0:
                    self._logger.log_warning(
                        f"Engine {engine_name} has no test split. Skipping."
                    )
                    continue

                X_test_raw = dataset.y[test_indices]
                y_test_raw = dataset.X[test_indices]

                X_test_norm = engine.transform_objective(X_test_raw)
                y_test_norm = engine.transform_decision(y_test_raw)

                n_test = len(X_test_norm)

                # Check for batch support (Neural Solvers: MDN, CVAE, INN)
                supports_batch = engine_type.upper() in ["MDN", "CVAE", "INN"]

                if supports_batch:
                    self._logger.log_info(f"Using batch generation for {engine_name}")
                    res = engine.solver.generate(
                        X_test_norm, n_samples=command.num_samples
                    )
                    all_candidates_X_norm = res.candidates_X.reshape(
                        n_test, command.num_samples, -1
                    )
                    all_candidates_y_norm = res.candidates_y.reshape(
                        n_test, command.num_samples, -1
                    )
                else:
                    all_candidates_X_norm = []
                    all_candidates_y_norm = []
                    for i in range(n_test):
                        target_y_norm = X_test_norm[i]
                        res = engine.solver.generate(
                            target_y_norm, n_samples=command.num_samples
                        )
                        all_candidates_X_norm.append(res.candidates_X)
                        all_candidates_y_norm.append(res.candidates_y)
                    all_candidates_X_norm = np.stack(all_candidates_X_norm)
                    all_candidates_y_norm = np.stack(all_candidates_y_norm)

                y_train_norm = engine.transform_objective(
                    dataset.y[dataset.train_indices]
                )

                # 1. Objective Space Assessment (Accuracy)
                objective_assessment = ObjectiveSpaceAuditor.audit(
                    training_objectives=y_train_norm,
                    candidates=all_candidates_y_norm,
                    reference=X_test_norm,
                    distance="euclidean",
                )

                # 2. Decision Space Assessment (Probabilistic/Calibration)
                capability = self.map_capability(engine_type)
                decision_assessment = DecisionSpaceAuditor.audit(
                    capability=capability,
                    samples=all_candidates_X_norm,
                    truth=y_test_norm,
                )

                # 3. Create Aggregate
                report = DiagnosticReport.create(
                    engine=EngineVO(
                        type=engine_type,
                        version=version,
                        mapping_direction=MappingDirection.INVERSE,
                        capability=capability,
                    ),
                    dataset_name=command.dataset_name,
                    num_samples=command.num_samples,
                    objective_space=objective_assessment,
                    decision_space=decision_assessment,
                )

                # Persist report
                # self._diag_repo.save(report)
                reports.append(report)

            # Structured sentinel for completion
            results_payload = {
                "event": "done",
                "status": "100% Complete",
                "progress": 100,
                "reports": [report.model_dump() for report in reports],
            }
            await publish_safe(results_payload)
            return Result.ok(reports)

        except FileNotFoundError as e:
            await publish_safe(
                {
                    "event": "error",
                    "status": "Error",
                    "message": "Dataset or engine not found",
                    "details": str(e),
                }
            )
            return Result.fail(
                message="Dataset or engine not found",
                details=str(e),
                code="NOT_FOUND",
            )
        except Exception as e:
            self._logger.log_error(f"Diagnostic failed: {str(e)}")
            await publish_safe(
                {
                    "event": "error",
                    "status": "Error",
                    "message": "Internal error",
                    "details": str(e),
                }
            )
            return Result.fail(
                message="Diagnostic failed",
                details=str(e),
                code="INTERNAL_ERROR",
            )

    def _initialize_engines(
        self,
        candidates: list[EngineCandidate],
        dataset_name: str,
    ) -> list[tuple[InverseMappingEngine, int]]:
        """
        Initializes engine instances from the repository.
        """
        engines = []

        for candidate in candidates:
            engine = self._engine_repository.load(
                dataset_name=dataset_name,
                solver_type=candidate.solver_type,
                version=candidate.version,
            )

            version = candidate.version
            if version is None:
                # Resolve version from list
                summaries = self._engine_repository.list_all(
                    dataset_name, candidate.solver_type
                )
                if summaries:
                    version = summaries[0].get("version", 0)

            engines.append((engine, version))

        return engines
