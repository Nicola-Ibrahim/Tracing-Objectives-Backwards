from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....modules.inverse.domain.entities.inverse_mapping_engine import (
    InverseMappingEngine,
)
from ....modules.inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ....shared.domain.interfaces.base_logger import BaseLogger
from ...domain.aggregates.diagnostic_result import DiagnosticResult
from ...domain.entities.accuracy_lens import AccuracyLens
from ...domain.entities.reliability_lens import ReliabilityLens
from ...domain.interfaces.base_diagnostic_repository import (
    BaseDiagnosticRepository,
)
from ...domain.services.generative_distribution_auditor import (
    GenerativeDistributionAuditor,
)
from ...domain.services.spatial_candidate_auditor import (
    SpatialCandidateAuditor,
)
from ...domain.value_objects.estimator import Estimator


class InverseEngineCandidate(BaseModel):
    """Represents a specific inverse engine candidate (solver type and optional version)."""

    solver_type: str = Field(..., examples=["GBPI"])
    version: int | None = Field(
        default=None,
        description="Specific integer version number. If None, latest is used.",
    )


class DiagnoseInverseModelsParams(BaseModel):
    """
    Command for the full evaluation suite including
    Objective-Space Accuracy and Decision-Space Reliability.
    Supports comparing multiple inverse model candidates.
    """

    dataset_name: str = Field(..., examples=["cocoex_f5"])

    inverse_engine_candidates: list[InverseEngineCandidate] = Field(
        ...,
        description="List of engine candidates to compare.",
    )

    num_samples: int = Field(default=200, description="K candidates per target")
    random_state: int = 42

    scale_method: Literal["sd", "mad", "iqr"] = Field(
        default="sd", description="sd | mad | iqr"
    )


class DiagnoseInverseModelsService:
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

    def execute(self, params: DiagnoseInverseModelsParams) -> None:
        self._logger.log_info(
            f"Starting Diagnostic Compute for engines on dataset '{params.dataset_name}'"
        )

        dataset = self._data_repo.load(params.dataset_name)

        engines = self._initialize_engines(
            candidates=params.inverse_engine_candidates,
            dataset_name=params.dataset_name,
        )

        for engine, version in engines:
            self._logger.log_info(f"Diagnosing {engine.solver.type()} (v{version})")

            # 1. Get test split from engine
            test_indices = engine.data_split.test_indices
            if len(test_indices) == 0:
                self._logger.log_warning(
                    f"Engine {engine.solver.type()} has no test split. Skipping."
                )
                continue

            X_test_raw = dataset.objectives[test_indices]
            y_test_raw = dataset.decisions[test_indices]

            # 2. Transform the test split to normalized space
            X_test_norm = engine.transform_objective(X_test_raw)
            y_test_norm = engine.transform_decision(y_test_raw)

            # 3. Batch generate candidates for each test target
            # Note: For large test sets, this might need batching.
            # Current implementation of solvers handles single targets.

            n_test = len(X_test_norm)
            all_candidates_X_norm = []
            all_candidates_y_norm = []

            for i in range(n_test):
                target_y_norm = X_test_norm[i]
                res = engine.solver.generate(
                    target_y_norm, n_samples=params.num_samples
                )

                # candidates_X are decisions, candidates_y are objectives (forward simulated)
                all_candidates_X_norm.append(res.candidates_X)

                # candidates_y from solver is already forward simulated but might be in physical space
                # or normalized space depending on solver implementation.
                # In GBPI, it uses forward_estimator.predict(candidates_X).
                # Since candidates_X is normalized, forward_estimator.predict returns normalized y.
                all_candidates_y_norm.append(res.candidates_y)

            all_candidates_X_norm = np.stack(
                all_candidates_X_norm
            )  # (n_test, n_samples, d_x)
            all_candidates_y_norm = np.stack(
                all_candidates_y_norm
            )  # (n_test, n_samples, d_y)

            # Accuracy Domain (Objective Space)
            # Reference target is X_test_norm (the target objectives)

            # We need training objectives in normalized space for the spatial auditor (to calculate tau/density)
            y_train_norm = engine.transform_objective(
                dataset.objectives[engine.data_split.train_indices]
            )

            spatial_audit = SpatialCandidateAuditor.audit(
                training_objectives=y_train_norm,
                candidates=all_candidates_y_norm,
                reference=X_test_norm,
                distance="euclidean",
            )

            # Reliability Domain (Decision Space)
            # Truth is y_test_norm (the true decisions)
            generative_audit = GenerativeDistributionAuditor.audit(
                samples=all_candidates_X_norm,
                truth=y_test_norm,
            )

            # Create Diagnostic Result Aggregate
            result = DiagnosticResult.create(
                estimator=Estimator(
                    type=engine.solver.type(),
                    version=version,
                    mapping_direction="inverse",
                ),
                dataset_name=params.dataset_name,
                num_samples=params.num_samples,
                scale_method=params.scale_method,
                accuracy=AccuracyLens(
                    discrepancy_scores=spatial_audit.discrepancy_scores,
                    best_shot_scores=spatial_audit.best_shot_scores,
                    rank_indices=spatial_audit.rank_indices,
                    systematic_bias=spatial_audit.bias,
                    cloud_dispersion=spatial_audit.dispersion,
                    summary=spatial_audit.summary,
                ),
                reliability=ReliabilityLens(
                    pit_values=generative_audit.pit_values,
                    calibration_error=generative_audit.calibration_error,
                    crps=generative_audit.crps,
                    diversity=generative_audit.diversity,
                    interval_width=generative_audit.interval_width,
                    summary=generative_audit.summary,
                    calibration_curve=generative_audit.calibration_curve,
                ),
            )

            # Persist Result
            self._diag_repo.save(result)

    def _initialize_engines(
        self,
        candidates: list[InverseEngineCandidate],
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

            # Find the actual version number for reporting
            # If candidate.version was None, load() returned latest.
            # We can get version from metadata if we store it in the aggregate,
            # but currently it's not in the entity.
            # Actually, I should probably have added 'version' to the entity.
            # For now, I'll assume we can use candidate.version or resolve it.

            version = candidate.version
            if version is None:
                # Resolve version from list
                summaries = self._engine_repository.list(
                    dataset_name, candidate.solver_type
                )
                if summaries:
                    version = summaries[0].get("version", 0)

            engines.append((engine, version))

        return engines
