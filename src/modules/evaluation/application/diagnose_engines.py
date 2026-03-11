from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...inverse.domain.entities.inverse_mapping_engine import (
    InverseMappingEngine,
)
from ...inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ...shared.domain.interfaces.base_logger import BaseLogger
from ...shared.result import Result
from ..domain.aggregates.diagnostic_result import DiagnosticResult
from ..domain.entities.accuracy_lens import AccuracyLens
from ..domain.entities.reliability_lens import ReliabilityLens
from ..domain.interfaces.base_diagnostic_repository import (
    BaseDiagnosticRepository,
)
from ..domain.services.accuracy_auditor import AccuracyAuditor
from ..domain.services.reliability_auditor import ReliabilityAuditor
from ..domain.value_objects.empirical_distribution import EmpiricalDistribution
from ..domain.value_objects.estimator import Estimator


class EngineCandidate(BaseModel):
    """
    Identifies a specific inverse engine and version for evaluation.
    This is an application-layer concern (part of the command).
    """

    solver_type: str = Field(..., examples=["GBPI"])
    version: int | None = Field(
        default=None,
        description="Specific integer version number. If None, latest is used.",
    )


class RunDiagnosticsCommand(BaseModel):
    """
    Command for the full evaluation suite including
    Objective-Space Accuracy and Decision-Space Reliability.
    Supports comparing multiple inverse model candidates.
    """

    dataset_name: str = Field(..., examples=["cocoex_f5"])

    inverse_engine_candidates: list[EngineCandidate] = Field(
        ...,
        description="List of engine candidates to compare.",
    )

    num_samples: int = Field(default=200, description="K candidates per target")
    random_state: int = 42

    scale_method: Literal["sd", "mad", "iqr"] = Field(
        default="sd", description="sd | mad | iqr"
    )


class RunDiagnosticsService:
    """
    Orchestrator for the evaluation suite.
    Computes diagnostics and persists them via the Diagnostic Repository.
    """

    def __init__(
        self,
        engine_repository: BaseInverseMappingEngineRepository,
        data_repository: BaseDatasetRepository,
        diagnostic_repository: BaseDiagnosticRepository,
        logger: BaseLogger,
    ):
        self._engine_repository = engine_repository
        self._data_repo = data_repository
        self._diag_repo = diagnostic_repository
        self._logger = logger

    def execute(self, command: RunDiagnosticsCommand) -> Result[list[DiagnosticResult]]:
        try:
            self._logger.log_info(
                f"Starting Diagnostic Compute for engines on dataset '{command.dataset_name}'"
            )

            dataset = self._data_repo.load(command.dataset_name)

            engines_to_diagnose = self._initialize_engines(
                candidates=command.inverse_engine_candidates,
                dataset_name=command.dataset_name,
            )

            diagnostics = []

            for engine, version in engines_to_diagnose:
                engine_name = f"{engine.solver.type()} (v{version})"
                self._logger.log_info(f"Diagnosing {engine_name}")

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

                accuracy_audit = AccuracyAuditor.audit(
                    training_objectives=y_train_norm,
                    candidates=all_candidates_y_norm,
                    reference=X_test_norm,
                    distance="euclidean",
                )

                reliability_audit = ReliabilityAuditor.audit(
                    samples=all_candidates_X_norm,
                    truth=y_test_norm,
                )

                result = DiagnosticResult.create(
                    estimator=Estimator(
                        type=engine.solver.type(),
                        version=version,
                        mapping_direction="inverse",
                    ),
                    dataset_name=command.dataset_name,
                    num_samples=command.num_samples,
                    scale_method=command.scale_method,
                    accuracy=AccuracyLens(
                        discrepancy_profile=EmpiricalDistribution.from_samples(
                            accuracy_audit.discrepancy_scores
                        ),
                        summary=accuracy_audit.summary,
                    ),
                    reliability=ReliabilityLens(
                        calibration_error=float(reliability_audit.calibration_error),
                        crps=float(reliability_audit.crps),
                        pit_profile=reliability_audit.calibration_curve,
                        calibration_curve=reliability_audit.calibration_curve,
                        summary=reliability_audit.summary,
                    ),
                )

                # Persist result
                self._diag_repo.save(result)
                diagnostics.append(result)

            return Result.ok(diagnostics)

        except FileNotFoundError as e:
            return Result.fail(
                message="Dataset or engine not found",
                details=str(e),
                code="NOT_FOUND",
            )
        except Exception as e:
            self._logger.log_error(f"Diagnostic failed: {str(e)}")
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
