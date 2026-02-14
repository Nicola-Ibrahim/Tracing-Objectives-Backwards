import numpy as np

from .....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from .....modeling.domain.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from .....modeling.domain.interfaces.base_repository import BaseModelArtifactRepository
from .....shared.domain.interfaces.base_logger import BaseLogger
from ....domain.aggregates.diagnostic_result import DiagnosticResult
from ....domain.entities.accuracy_lens import AccuracyLens
from ....domain.entities.reliability_lens import ReliabilityLens
from ....domain.interfaces.base_diagnostic_repository import (
    BaseDiagnosticRepository,
)
from ....domain.services.generative_distribution_auditor import (
    GenerativeDistributionAuditor,
)
from ....domain.services.spatial_candidate_auditor import (
    SpatialCandidateAuditor,
)
from ....domain.value_objects.estimator import Estimator
from .command import DiagnoseInverseModelsCommand, InverseEstimatorCandidate


class DiagnoseInverseModelsHandler:
    """
    Orchestrator for the Thesis-aligned evaluation suite.
    Computes diagnostics and persists them via the Diagnostic Repository.
    """

    def __init__(
        self,
        model_artifact_repository: BaseModelArtifactRepository,
        data_repository: BaseDatasetRepository,
        diagnostic_repository: BaseDiagnosticRepository,
        logger: BaseLogger,
    ):
        self._model_artifact_repo = model_artifact_repository
        self._data_repo = data_repository
        self._diag_repo = diagnostic_repository
        self._logger = logger

    def execute(self, command: DiagnoseInverseModelsCommand) -> dict[str, int]:
        self._logger.log_info(
            f"Starting Diagnostic Compute for: "
            f"{[(c.type.value, c.version) for c in command.inverse_estimator_candidates]}"
        )

        # 1. Load context
        dataset = self._data_repo.load(command.dataset_name)

        # Load Forward Estimator
        forward_estimator = self._model_artifact_repo.get_latest_version(
            command.forward_estimator_type.value, "forward", command.dataset_name
        ).estimator

        # 3. Initialize inverse estimators
        inverse_estimators = self._initialize_estimators(
            inverse_estimator_candidates=command.inverse_estimator_candidates,
            dataset_name=command.dataset_name,
        )

        # 4. Run diagnostic workflow
        for version, inverse_estimator in inverse_estimators:
            # Sampling
            y_test_norm = dataset.processed.objectives_test

            samples_X_norm = inverse_estimator.sample(
                y_test_norm, n_samples=command.num_samples
            )

            if samples_X_norm.ndim == 2:
                samples_X_norm = samples_X_norm[:, np.newaxis, :]

            n_test, k_samples, x_dim = samples_X_norm.shape

            # Reshape the X-normalized-space samples to a flat array
            samples_X_norm_flat = samples_X_norm.reshape(-1, x_dim)

            # Denormalize the X-normalized-space samples
            samples_X_phys = dataset.processed.decisions_normalizer.inverse_transform(
                samples_X_norm_flat
            )

            # Forward simulate the X-space samples
            pred_obj_phys = forward_estimator.predict(samples_X_phys)

            # Normalize the y-space predictions
            pred_obj_norm = dataset.processed.objectives_normalizer.transform(
                pred_obj_phys
            )

            # Reshape the y-space predictions to match the test set shape
            y_pred = pred_obj_norm.reshape(n_test, k_samples, -1)

            # Accuracy Domain (Objective Space - Standardized Lens)
            y_target = dataset.processed.objectives_test

            # (Objective Space - Standardized Lens)
            spatial_audit = SpatialCandidateAuditor.audit(
                training_objectives=dataset.processed.objectives_train,
                candidates=y_pred,
                reference=y_target,
                distance="euclidean",
            )

            # Reliability Domain (Decision Space - Probabilistic Lens)
            x_truth = dataset.processed.decisions_test
            generative_audit = GenerativeDistributionAuditor.audit(
                samples=samples_X_norm,
                truth=x_truth,
            )

            # Create Diagnostic Result Aggregate
            result = DiagnosticResult.create(
                estimator=Estimator(
                    type=inverse_estimator.type,
                    version=version,
                    mapping_direction="inverse",
                ),
                dataset_name=command.dataset_name,
                num_samples=command.num_samples,
                scale_method=command.scale_method,
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

    def _initialize_estimators(
        self,
        inverse_estimator_candidates: list[InverseEstimatorCandidate],
        dataset_name: str,
    ) -> list[tuple[str, int], BaseEstimator]:
        """
        Initializes inverse estimator instances from the repository.
        """
        inverse_estimators: list[tuple[str, int], BaseEstimator] = []

        for candidate in inverse_estimator_candidates:
            inverse_type = candidate.type.value
            version = candidate.version

            # Resolve from repository
            artifact = self._model_artifact_repo.get_version_by_number(
                estimator_type=inverse_type,
                version=version,
                mapping_direction="inverse",
                dataset_name=dataset_name,
            )

            # Sanity check for probabilistic properties
            if not isinstance(artifact.estimator, ProbabilisticEstimator):
                self._logger.log_info(
                    f"Estimator {inverse_type} (v{version}) is not probabilistic. Statistical metrics might be unreliable."
                )

            inverse_estimators.append((version, artifact.estimator))

        return inverse_estimators
