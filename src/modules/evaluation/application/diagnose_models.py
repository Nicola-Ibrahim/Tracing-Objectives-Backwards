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
from ..domain.aggregates.diagnostic_result import DiagnosticResult
from ..domain.entities.accuracy_lens import AccuracyLens
from ..domain.entities.reliability_lens import ReliabilityLens
from ..domain.interfaces.base_diagnostic_repository import (
    BaseDiagnosticRepository,
)
from ..domain.services.generative_distribution_auditor import (
    GenerativeDistributionAuditor,
)
from ..domain.services.spatial_candidate_auditor import (
    SpatialCandidateAuditor,
)
from ..domain.value_objects.estimator import Estimator


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

    def execute(self, params: DiagnoseInverseModelsParams) -> dict:
        self._logger.log_info(
            f"Starting Diagnostic Compute for engines on dataset '{params.dataset_name}'"
        )

        dataset = self._data_repo.load(params.dataset_name)

        engines_to_diagnose = self._initialize_engines(
            candidates=params.inverse_engine_candidates,
            dataset_name=params.dataset_name,
        )

        ecdf_results = {}
        pit_results = {}
        mace_results = {}
        engine_names = []
        warnings = []

        for engine, version in engines_to_diagnose:
            engine_name = f"{engine.solver.type()} (v{version})"
            engine_names.append(engine_name)
            self._logger.log_info(f"Diagnosing {engine_name}")

            # 1. Get test split from engine
            test_indices = engine.data_split.test_indices
            if len(test_indices) == 0:
                warnings.append(f"Engine {engine_name} has no test split. Skipping.")
                continue

            X_test_raw = dataset.objectives[test_indices]
            y_test_raw = dataset.decisions[test_indices]

            # 2. Transform the test split to normalized space
            X_test_norm = engine.transform_objective(X_test_raw)
            y_test_norm = engine.transform_decision(y_test_raw)

            # 3. Batch generate candidates for each test target
            n_test = len(X_test_norm)
            all_candidates_X_norm = []
            all_candidates_y_norm = []

            for i in range(n_test):
                target_y_norm = X_test_norm[i]
                res = engine.solver.generate(
                    target_y_norm, n_samples=params.num_samples
                )
                all_candidates_X_norm.append(res.candidates_X)
                all_candidates_y_norm.append(res.candidates_y)

            all_candidates_X_norm = np.stack(all_candidates_X_norm)
            all_candidates_y_norm = np.stack(all_candidates_y_norm)

            # Accuracy Domain (Objective Space)
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
                    discrepancy_scores=spatial_audit.discrepancy_scores.tolist(),
                    best_shot_scores=spatial_audit.best_shot_scores.tolist(),
                    rank_indices=spatial_audit.rank_indices.tolist(),
                    systematic_bias=float(spatial_audit.bias),
                    cloud_dispersion=float(spatial_audit.dispersion),
                    summary=spatial_audit.summary,
                ),
                reliability=ReliabilityLens(
                    pit_values=generative_audit.pit_values.tolist(),
                    calibration_error=float(generative_audit.calibration_error),
                    crps=float(generative_audit.crps),
                    diversity=float(generative_audit.diversity),
                    interval_width=float(generative_audit.interval_width),
                    summary=generative_audit.summary,
                    calibration_curve=generative_audit.calibration_curve,
                ),
            )

            # Persist Result
            self._diag_repo.save(result)

            # Format for Response
            ecdf_results[engine_name] = self._calculate_ecdf(
                spatial_audit.discrepancy_scores
            )
            # PIT values represent cumulative density [0, 1]. ECDF of PIT values allows for calibration plotting (Ideal is x=y).
            pit_results[engine_name] = self._calculate_ecdf(generative_audit.pit_values)
            mace_results[engine_name] = float(generative_audit.calibration_error)

        return {
            "dataset_name": params.dataset_name,
            "engines": engine_names,
            "ecdf": ecdf_results,
            "pit": pit_results,
            "mace": mace_results,
            "warnings": warnings,
        }

    def get_cached_diagnostics(
        self, params: DiagnoseInverseModelsParams
    ) -> dict | None:
        """
        Retrieves formatted diagnostic results from the repository if they exist for ALL engines.
        Returns formatted result dict or None if any engine is missing.
        """
        ecdf_results = {}
        pit_results = {}
        mace_results = {}
        engine_names = []
        warnings = []

        try:
            # Resolve actual versions first
            engines_to_check = self._initialize_engines(
                candidates=params.inverse_engine_candidates,
                dataset_name=params.dataset_name,
            )

            for engine, version in engines_to_check:
                engine_name = f"{engine.solver.type()} (v{version})"
                engine_names.append(engine_name)

                # Try to load latest run
                try:
                    cached = self._diag_repo.get_latest_run(
                        estimator_type=engine.solver.type(),
                        estimator_version=version,
                        dataset_name=params.dataset_name,
                    )

                    # Found it, extract metrics
                    ecdf_results[engine_name] = self._calculate_ecdf(
                        np.array(cached.accuracy.discrepancy_scores)
                    )
                    pit_results[engine_name] = self._calculate_ecdf(
                        np.array(cached.reliability.pit_values)
                    )
                    mace_results[engine_name] = float(
                        cached.reliability.calibration_error
                    )

                except (FileNotFoundError, IndexError):
                    # Cache miss for this specific engine
                    return None

            return {
                "dataset_name": params.dataset_name,
                "engines": engine_names,
                "ecdf": ecdf_results,
                "pit": pit_results,
                "mace": mace_results,
                "warnings": warnings,
            }
        except Exception as e:
            self._logger.log_error(f"Error checking diagnostic cache: {str(e)}")
            return None

    def _calculate_ecdf(self, data: np.ndarray) -> dict:
        """Helper to calculate Empirical Cumulative Distribution Function points."""
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        # Resample to 100 points if too large for bandwidth
        if len(x) > 100:
            indices = np.linspace(0, len(x) - 1, 100).astype(int)
            x = x[indices]
            y = y[indices]
        return {"x": x.tolist(), "y": y.tolist()}

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
