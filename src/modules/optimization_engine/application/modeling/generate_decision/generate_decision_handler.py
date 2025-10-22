import numpy as np

from ....domain.assurance.decision_validation.interfaces import (
    BaseDecisionValidationCalibrationRepository,
)
from ....domain.assurance.decision_validation.services.decision_validation_service import (
    DecisionValidationService,
)
from ....domain.assurance.feasibility.errors import ObjectiveOutOfBoundsError
from ....domain.assurance.feasibility.services.objective_feasibility_service import (
    ObjectiveFeasibilityService,
)
from ....domain.assurance.feasibility.value_objects.pareto_front import ParetoFront
from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ....domain.modeling.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ...factories.assurance import DiversityStrategyFactory, ScoreStrategyFactory
from .generate_decision_command import GenerateDecisionCommand


class GenerateDecisionCommandHandler:
    """Generate y for a requested x target and apply assurance checks."""

    def __init__(
        self,
        model_repository: BaseModelArtifactRepository,
        processed_data_repository: BaseDatasetRepository,
        logger: BaseLogger,
        calibration_repository: BaseDecisionValidationCalibrationRepository,
    ):
        """Wire infrastructure dependencies and optional assurance services."""
        self._model_repository = model_repository
        self._processed_data_repo = processed_data_repository
        self._logger = logger
        self._feasibility_service = ObjectiveFeasibilityService(
            scorer=ScoreStrategyFactory().create("kde"),
            diversity_registry=DiversityStrategyFactory().create_bunch(["kmeans"]),
        )
        dv_calibration = calibration_repository.load()
        self._decision_validation_service = DecisionValidationService(
            tolerance=0.03,
            ood_calibrator=dv_calibration.ood_calibrator,
            conformal_calibrator=dv_calibration.conformal_calibrator,
        )
        self._calibration_repository = calibration_repository

    def execute(self, command: GenerateDecisionCommand) -> np.ndarray:
        """Generate y for the requested estimator and x target, then validate."""
        estimator = self._load_estimator(command.estimator_type)
        processed: ProcessedDataset = self._processed_data_repo.load("dataset")

        objectives_normalizer = processed.X_normalizer  # objective normalizer
        decision_normalizer = processed.y_normalizer  # decision normalizer

        pareto_set = processed.pareto.set
        pareto_front = processed.pareto.front

        target_objective_X, target_objective_X_norm = self._build_target_X(
            command.target_objective, objectives_normalizer
        )
        self._logger.log_info(f"Target x (raw): {target_objective_X}")
        self._logger.log_info(f"Target x (normalized): {target_objective_X_norm}")

        # NOTE: Comment out feasibility for now as it is not yet used in practice
        # if command.feasibility_enabled:
        #     self._validate_feasibility(
        #         pareto_front=pareto_front,
        #         objectives_normalizer=objectives_normalizer,
        #         target_x=target_x,
        #         distance_tolerance=command.distance_tolerance,
        #         num_suggestions=command.num_suggestions,
        #         diversity_method=command.diversity_method,
        #         suggestion_noise_scale=command.suggestion_noise_scale,
        #     )

        generated_decision_y, generated_decision_y_norm = self.infer(
            estimator=estimator,
            X_norm=target_objective_X_norm,
            normalizer=decision_normalizer,
        )

        self._logger.log_info(f"Generated y (raw): {generated_decision_y}")
        self._logger.log_info(f"Generated y (normalized): {generated_decision_y_norm}")

        if command.validation_enabled:
            self._run_decision_validation(
                X=generated_decision_y_norm,
                y_target=target_objective_X_norm,
            )

        return generated_decision_y

    def _load_estimator(self, estimator_type: EstimatorTypeEnum) -> BaseEstimator:
        """Return the most recent estimator artifact matching the type."""
        artifact = self._model_repository.get_latest_version(
            estimator_type=estimator_type.value
        )
        return artifact.estimator

    def _build_target_X(self, target_objective_X: np.ndarray, normalizer):
        """Return target objective in raw and normalized space."""
        target_objective_X = np.asarray(target_objective_X, dtype=float)
        target_objective_X_norm = normalizer.transform(
            np.atleast_2d(target_objective_X)
        )[0]
        return target_objective_X, target_objective_X_norm

    def _validate_feasibility(
        self,
        *,
        pareto_front: np.ndarray | None,
        objectives_normalizer,
        target_x: tuple[np.ndarray, np.ndarray],
        distance_tolerance: float,
        num_suggestions: int,
        diversity_method: str,
        suggestion_noise_scale: float,
    ) -> None:
        """Check that target x fits within the support of the Pareto front."""
        if pareto_front is None:
            return

        pareto_front_norm = objectives_normalizer.transform(pareto_front)
        service = self._feasibility_service

        pareto_vo = ParetoFront(
            raw=pareto_front,
            normalized=pareto_front_norm,
        )

        try:
            service.validate(
                pareto_front=pareto_vo,
                tolerance=distance_tolerance,
                target=target_x,
                num_suggestions=num_suggestions,
                suggestion_noise_scale=suggestion_noise_scale,
                diversity_method=diversity_method,
            )
        except ObjectiveOutOfBoundsError as error:
            self._handle_feasibility_error(error, objectives_normalizer)

    def infer(
        self,
        *,
        estimator: BaseEstimator,
        X_norm: np.ndarray,
        normalizer,
        n_samples: int | None = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return denormalized and normalized outputs for the provided inputs."""

        X_input = np.atleast_2d(np.asarray(X_norm, dtype=float))

        if isinstance(estimator, ProbabilisticEstimator):
            sample_count = int(n_samples) if n_samples is not None else 256
            y_norm = estimator.predict_mean(X_input, n_samples=sample_count)
        else:
            y_norm = estimator.predict(X_input)

        y_norm = np.atleast_2d(np.asarray(y_norm, dtype=float))
        if y_norm.shape[0] != X_input.shape[0]:
            raise ValueError(
                "Estimator returned mismatched batch size during inference."
            )

        y_raw = normalizer.inverse_transform(y_norm)[0]
        return y_raw, y_norm[0]

    def _handle_feasibility_error(
        self, error: ObjectiveOutOfBoundsError, objectives_normalizer
    ) -> None:
        """Report infeasible x diagnostics and re-raise the originating error."""
        self._logger.log_error(
            f"Feasibility failed: {error.reason.value} | {error.message}"
        )
        if error.suggestions is None:
            raise error

        suggestions = np.atleast_2d(np.asarray(error.suggestions))
        if suggestions.size == 0:
            raise error

        try:
            denorm_suggestions = objectives_normalizer.inverse_transform(suggestions)
        except Exception:
            raise error

        suggestions_str = "; ".join(str(value.tolist()) for value in denorm_suggestions)
        self._logger.log_error(f"Suggestions (original scale): {suggestions_str}")
        raise error

    def _run_decision_validation(self, X: np.ndarray, y_target: np.ndarray) -> None:
        """Execute decision validation service if configured."""

        report = self._decision_validation_service.validate(X=X, y_target=y_target)

        if self._logger:
            summary = {
                "passed": report.verdict.value,
                "gate1": {
                    "md2": report.metrics.get("gate1_md2"),
                    "thr": report.metrics.get("gate1_md2_threshold"),
                },
                "gate2": {
                    "q": report.metrics.get("gate2_conformal_radius_q"),
                    "dist": report.metrics.get("gate2_dist_to_target_l2"),
                },
            }

            self._logger.log_metrics(summary)
            explanations = {gate.name: gate.explanation for gate in report.gate_results}
            self._logger.log_info(
                f"[assurance] Verdict={report.verdict.value} | Reasons={explanations}"
            )
