import numpy as np

from .....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from .....modeling.domain.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from .....modeling.domain.interfaces.base_repository import BaseModelArtifactRepository
from .....shared.domain.interfaces.base_logger import BaseLogger
from ....domain.accuracy import scaling
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
from ....domain.value_objects.accuracy_summary import AccuracySummary
from ....domain.value_objects.calibration_curve import CalibrationCurve
from ....domain.value_objects.estimator import Estimator
from ....domain.value_objects.reliability_summary import ReliabilitySummary
from ....domain.value_objects.spatial_candidates import SpatialCandidates
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

        # 2. Extract Scaling τ from training marginals
        # TODO: Move to be factory
        tau = self._get_scale_vector(
            vector=dataset.processed.objectives_train,
            method=command.scale_method,
        )

        self._logger.log_info(f"Scale τ ({command.scale_method}): {tau.tolist()}")

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

            spatial_audit = SpatialCandidateAuditor.audit(
                candidates=y_pred,
                reference=y_target,
                tau=tau,
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
                    best_shot_residuals=np.min(
                        spatial_audit.discrepancy_scores, axis=1
                    ),
                    systematic_bias=spatial_audit.bias,
                    cloud_dispersion=spatial_audit.dispersion,
                    summary=AccuracySummary(
                        mean_best_shot=float(
                            np.mean(np.min(spatial_audit.discrepancy_scores, axis=1))
                        ),
                        median_best_shot=float(
                            np.median(np.min(spatial_audit.discrepancy_scores, axis=1))
                        ),
                        mean_bias=float(np.mean(spatial_audit.bias)),
                        mean_dispersion=float(np.mean(spatial_audit.dispersion)),
                    ),
                ),
                reliability=ReliabilityLens(
                    pit_values=generative_audit.pit_values,
                    calibration_error=generative_audit.calibration_error,
                    crps=generative_audit.crps,
                    diversity=generative_audit.diversity,
                    interval_width=generative_audit.interval_width,
                    summary=ReliabilitySummary(
                        mean_crps=generative_audit.crps,
                        mean_diversity=float(np.mean(generative_audit.diversity)),
                        mean_interval_width=float(
                            np.mean(generative_audit.interval_width)
                        ),
                    ),
                    calibration_curve=CalibrationCurve(
                        pit_values=np.sort(generative_audit.pit_values),
                        cdf_y=np.arange(1, len(generative_audit.pit_values) + 1)
                        / len(generative_audit.pit_values),
                    ),
                ),
                candidates=SpatialCandidates(
                    ordered=samples_X_norm[
                        np.arange(n_test)[:, np.newaxis], spatial_audit.rank_indices
                    ],
                    median=np.median(samples_X_norm, axis=1),
                    std=np.std(samples_X_norm, axis=1),
                ),
            )

            # Persist Result
            self._diag_repo.save(result)

    def _get_scale_vector(self, vector: np.ndarray, method: str) -> np.ndarray:
        if method == "sd":
            return scaling.compute_sd_scale(vector)
        if method == "mad":
            return scaling.compute_mad_scale(vector)
        if method == "iqr":
            return scaling.compute_iqr_scale(vector)
        raise ValueError(f"Unknown scale method: {method}")

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
