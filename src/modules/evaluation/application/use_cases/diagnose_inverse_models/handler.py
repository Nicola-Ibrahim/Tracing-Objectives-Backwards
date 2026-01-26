import numpy as np

from .....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from .....modeling.domain.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from .....modeling.domain.interfaces.base_repository import BaseModelArtifactRepository
from .....shared.domain.interfaces.base_logger import BaseLogger
from ....domain.accuracy import plausibility, residuals, scaling
from ....domain.entities.diagnostic_result import (
    AccuracyLens,
    AccuracySummary,
    CalibrationCurve,
    DiagnosticResult,
    ReliabilityLens,
    ReliabilitySummary,
    SpatialCandidates,
)
from ....domain.interfaces.base_diagnostic_repository import (
    BaseDiagnosticRepository,
)
from ....domain.reliability import calibration, distribution_stats
from .command import DiagnoseInverseModelsCommand, InverseEstimatorCandidate


class DiagnoseInverseModelsHandler:
    """
    Orchestrator for the Thesis-aligned evaluation suite.
    Computes diagnostics and persists them via the Diagnostic Repository.
    """

    def __init__(
        self,
        model_repository: BaseModelArtifactRepository,
        data_repository: BaseDatasetRepository,
        diagnostic_repository: BaseDiagnosticRepository,
        logger: BaseLogger,
    ):
        self._model_repo = model_repository
        self._data_repo = data_repository
        self._diag_repo = diagnostic_repository
        self._logger = logger

    def execute(self, command: DiagnoseInverseModelsCommand) -> dict[str, int]:
        self._logger.log_info(
            f"Starting Diagnostic Compute for: "
            f"{[(c.type.value, c.version) for c in command.candidates]}"
        )

        # 1. Load context
        dataset = self._data_repo.load(command.dataset_name)
        if not dataset.processed:
            raise ValueError("Dataset has no processed data.")

        proc = dataset.processed

        # Load Forward Estimator
        forward_art = self._model_repo.get_latest_version(
            command.forward_estimator_type.value, "forward", command.dataset_name
        )
        forward_estimator = forward_art.estimator

        # 2. Extract Scaling τ from training marginals
        tau = self._get_scale_vector(proc.objectives_train, command.scale_method)
        self._logger.log_info(f"Scale τ ({command.scale_method}): {tau.tolist()}")

        # 3. Initialize inverse estimators
        inverse_estimators = self._initialize_estimators(
            candidates=command.candidates, dataset_name=command.dataset_name
        )

        # 4. Run diagnostic workflow
        run_registration = {}
        for display_name, inverse_estimator in inverse_estimators.items():
            # Get version for metadata
            # display_name format: "type (v1)"
            type_str = display_name.split(" (")[0]
            version_str = display_name.split("(v")[1].replace(")", "")
            version = int(version_str) if version_str != "latest" else -1

            self._logger.log_info(f"Computing diagnostics for {display_name}...")

            # Sampling
            y_test_norm = proc.objectives_test
            samples = inverse_estimator.sample(
                y_test_norm, n_samples=command.num_samples
            )
            if samples.ndim == 2:
                samples = samples[:, np.newaxis, :]

            # Forward Simulate
            n_test, k_samples, x_dim = samples.shape
            samples_flat = samples.reshape(-1, x_dim)
            samples_phys = proc.decisions_normalizer.inverse_transform(samples_flat)

            pred_obj_phys = forward_estimator.predict(samples_phys)
            pred_obj_norm = proc.objectives_normalizer.transform(pred_obj_phys)
            y_pred = pred_obj_norm.reshape(n_test, k_samples, -1)

            # Accuracy Domain (Objective Space - Standardized Lens)
            y_target = proc.objectives_test
            z_resid = residuals.compute_z_residuals(y_pred, y_target, tau)
            s_scores = residuals.compute_discrepancy_scores(z_resid)

            z_bar = plausibility.compute_mean_residual_vector(z_resid)
            bias_b = plausibility.compute_systematic_bias(z_bar)
            disp_v = plausibility.compute_cloud_dispersion(z_resid, z_bar)

            scenarios = [
                plausibility.classify_scenario(
                    b, v, command.bias_threshold, command.dispersion_threshold
                )
                for b, v in zip(bias_b, disp_v)
            ]

            # Reliability Domain (Decision Space - Probabilistic Lens)
            x_truth = proc.decisions_test
            pit_values = calibration.compute_pit_values(samples, x_truth)
            mace = calibration.compute_calibration_error(pit_values)
            crps = calibration.compute_crps(samples, x_truth)

            diversity = distribution_stats.compute_diversity(samples)
            intervals = distribution_stats.compute_interval_width(samples)

            # Spatial Information (Ranked candidates)
            ranked_candidates = self._rank_candidates_by_residuals(
                sampled_candidates=samples,
                residuals=s_scores,
            )

            # Create Diagnostic Result Aggregate
            result = DiagnosticResult.create(
                estimator_type=type_str,
                estimator_version=version,
                dataset_name=command.dataset_name,
                num_samples=command.num_samples,
                scale_method=command.scale_method,
                accuracy=AccuracyLens(
                    discrepancy_scores=s_scores,
                    best_shot_residuals=np.min(s_scores, axis=1),
                    systematic_bias=bias_b,
                    cloud_dispersion=disp_v,
                    scenarios=[s.value for s in scenarios],
                    summary=AccuracySummary(
                        mean_best_shot=float(np.mean(np.min(s_scores, axis=1))),
                        median_best_shot=float(np.median(np.min(s_scores, axis=1))),
                        mean_bias=float(np.mean(bias_b)),
                        mean_dispersion=float(np.mean(disp_v)),
                    ),
                ),
                reliability=ReliabilityLens(
                    pit_values=pit_values,
                    calibration_error=mace,
                    crps=crps,
                    diversity=diversity,
                    interval_width=intervals,
                    summary=ReliabilitySummary(
                        mean_crps=crps,
                        mean_diversity=float(np.mean(diversity)),
                        mean_interval_width=float(np.mean(intervals)),
                    ),
                    calibration_curve=CalibrationCurve(
                        pit_values=np.sort(pit_values),
                        cdf_y=np.arange(1, len(pit_values) + 1) / len(pit_values),
                    ),
                ),
                candidates=SpatialCandidates(
                    ordered=ranked_candidates,
                    median=np.median(samples, axis=1),
                    std=np.std(samples, axis=1),
                ),
            )

            # Persist Result
            run_number = self._diag_repo.save(result)
            run_registration[display_name] = run_number
            self._logger.log_info(f"Saved {display_name} as run {run_number}.")

        return run_registration

    def _get_scale_vector(self, y_train: np.ndarray, method: str) -> np.ndarray:
        if method == "sd":
            return scaling.compute_sd_scale(y_train)
        if method == "mad":
            return scaling.compute_mad_scale(y_train)
        if method == "iqr":
            return scaling.compute_iqr_scale(y_train)
        raise ValueError(f"Unknown scale method: {method}")

    def _initialize_estimators(
        self, candidates: list[InverseEstimatorCandidate], dataset_name: str
    ) -> dict[str, BaseEstimator]:
        """
        Initializes inverse estimator instances from the repository.
        """
        inverse_estimators: dict[str, BaseEstimator] = {}

        for candidate in candidates:
            inverse_type = candidate.type.value
            version = candidate.version
            display_name = (
                f"{inverse_type} (v{version})"
                if version
                else f"{inverse_type} (latest)"
            )

            # Resolve from repository
            artifact = self._model_repo.get_version_by_number(
                estimator_type=inverse_type,
                version=version,
                mapping_direction="inverse",
                dataset_name=dataset_name,
            )

            # Sanity check for probabilistic properties
            if not isinstance(artifact.estimator, ProbabilisticEstimator):
                self._logger.log_info(
                    f"Estimator {display_name} is not probabilistic. Statistical metrics might be unreliable."
                )

            inverse_estimators[display_name] = artifact.estimator

        return inverse_estimators

    def _rank_candidates_by_residuals(
        self, sampled_candidates: np.ndarray, residuals: np.ndarray
    ) -> np.ndarray:
        """
        Sorts generated candidates for each test point from closest to farthest from target.
        """
        sort_indices = np.argsort(residuals, axis=1)
        n_test_samples, n_samples, x_dim = sampled_candidates.shape
        row_indices = np.arange(n_test_samples)[:, np.newaxis]
        return sampled_candidates[row_indices, sort_indices]
