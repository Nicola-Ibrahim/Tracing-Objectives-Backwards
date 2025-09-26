from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from .....domain.modeling.interfaces.base_estimator import BaseEstimator
from ....domain.assurance.decision_validation import DecisionValidationService
from ....domain.assurance.decision_validation.interfaces import (
    DecisionValidationCalibrationRepository,
)
from ....domain.assurance.decision_validation.interfaces.base_conformal_calibrator import (
    BaseConformalCalibrator,
)
from ....domain.assurance.decision_validation.interfaces.base_ood_calibrator import (
    BaseOODCalibrator,
)
from ....domain.assurance.decision_validation.enums.verdict import Verdict
from ....domain.assurance.feasibility import (
    ObjectiveFeasibilityService,
)
from ....domain.assurance.feasibility.errors import ObjectiveOutOfBoundsError
from ....domain.assurance.feasibility.value_objects import ObjectiveVector, ParetoFront
from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from .generate_decision_command import GenerateDecisionCommand


@dataclass
class GeneratedY:
    """Normalized and denormalized forms of a generated y suggestion."""

    normalized: np.ndarray
    denormalized: np.ndarray


DEFAULT_DIVERSITY_METHOD = "euclidean"


class GenerateDecisionCommandHandler:
    """Generate y for a requested x target and apply assurance checks."""

    def __init__(
        self,
        model_repository: BaseModelArtifactRepository,
        processed_data_repository: BaseDatasetRepository,
        logger: BaseLogger,
        feasibility_service: ObjectiveFeasibilityService,
        decision_validation_service: DecisionValidationService,
        calibration_repository: DecisionValidationCalibrationRepository,
    ):
        """Wire infrastructure dependencies and optional assurance services."""
        self._model_repository = model_repository
        self._processed_data_repo = processed_data_repository
        self._logger = logger
        self._feasibility_service = feasibility_service
        self._dv_service = decision_validation_service
        self._dv_calibration_repository = calibration_repository
        self._dv_calibrators: dict[
            str, tuple[BaseOODCalibrator, BaseConformalCalibrator]
        ] = {}

    def execute(self, command: GenerateDecisionCommand) -> np.ndarray:
        """Generate y for the requested estimator and x target, then validate."""
        estimator = self._load_estimator(command.estimator_type)
        processed: ProcessedDataset = self._processed_data_repo.load("dataset")
        x_normalizer = processed.X_normalizer
        y_normalizer = processed.y_normalizer
        if processed.X_train is None or processed.y_train is None:
            raise AttributeError(
                "Processed dataset must provide X_train and y_train for assurance calibration"
            )
        pareto_y = processed.pareto.set
        pareto_front = processed.pareto.front

        target_x = self._build_x_vector(command.target_objective, x_normalizer)
        diversity_method = getattr(
            command, "diversity_method", DEFAULT_DIVERSITY_METHOD
        )

        suggestion_noise_scale = getattr(command, "suggestion_noise_scale", 0.05)

        if command.feasibility_enabled:
            self._validate_feasibility(
                pareto_front=pareto_front,
                x_normalizer=x_normalizer,
                target_x=target_x,
                distance_tolerance=command.distance_tolerance,
                num_suggestions=command.num_suggestions,
                diversity_method=diversity_method,
                suggestion_noise_scale=suggestion_noise_scale,
            )

        generated_y = self._generate_y(
            estimator=estimator,
            target_x_norm=target_x.normalized,
            y_normalizer=y_normalizer,
        )

        self._log_generation_metrics(
            y_raw=generated_y.denormalized,
            target_x_raw=target_x.raw,
            pareto_y=pareto_y,
        )

        if command.validation_enabled:
            self._run_decision_validation(
                y_norm=generated_y.normalized,
                x_target_norm=target_x.normalized,
                scope=command.estimator_type.value,
            )

        return generated_y.denormalized

    def _load_estimator(self, estimator_type: EstimatorTypeEnum) -> BaseEstimator:
        """Return the most recent estimator artifact matching the type."""
        artifact = self._model_repository.get_latest_version(
            estimator_type=estimator_type.value
        )
        return artifact.estimator

    def _build_x_vector(self, target_x: Sequence[float], normalizer) -> ObjectiveVector:
        """Create an objective vector wrapper for raw and normalized x."""
        target_array = np.asarray(target_x, dtype=float)
        self._logger.log_info(f"Target x (raw): {target_array}")
        target_norm = normalizer.transform(target_array)
        self._logger.log_info(f"Target x (normalized): {target_norm}")
        return ObjectiveVector(raw=target_array, normalized=target_norm)

    def _validate_feasibility(
        self,
        *,
        pareto_front: np.ndarray | None,
        x_normalizer,
        target_x: ObjectiveVector,
        distance_tolerance: float,
        num_suggestions: int,
        diversity_method: str,
        suggestion_noise_scale: float,
    ) -> None:
        """Check that target x fits within the support of the Pareto front."""
        if pareto_front is None:
            return

        pareto_front_norm = x_normalizer.transform(pareto_front)
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
            self._handle_feasibility_error(error, x_normalizer)

    def _handle_feasibility_error(
        self, error: ObjectiveOutOfBoundsError, x_normalizer
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
            denorm_suggestions = x_normalizer.inverse_transform(suggestions)
        except Exception:
            raise error

        suggestions_str = "; ".join(str(value.tolist()) for value in denorm_suggestions)
        self._logger.log_error(f"Suggestions (original scale): {suggestions_str}")
        raise error

    def _generate_y(
        self,
        *,
        estimator: BaseEstimator,
        target_x_norm: np.ndarray,
        y_normalizer,
    ) -> GeneratedY:
        """Generate y from the estimator and denormalize to the original scale."""
        y_norm = np.atleast_2d(estimator.predict(target_x_norm))
        self._logger.log_info(f"Generated y (normalized): {y_norm[0]}")
        y_raw = y_normalizer.inverse_transform(y_norm)[0]
        self._logger.log_info(f"Generated y (raw): {y_raw}")
        return GeneratedY(normalized=y_norm, denormalized=y_raw)

    def _log_generation_metrics(
        self,
        *,
        y_raw: np.ndarray,
        target_x_raw: np.ndarray,
        pareto_y: np.ndarray | None,
    ) -> None:
        """Log metrics describing y proximity to stored Pareto data."""
        metrics: dict[str, float] = {}

        # try:
        #     distances = np.linalg.norm(pareto_y - y_raw, axis=1)
        # except Exception:
        #     distances = None
        # if distances is not None:
        #     metrics["y_to_pareto_min_distance"] = float(np.min(distances))

        # if metrics:
        #     self._logger.log_metrics(metrics)

    def _run_decision_validation(
        self,
        *,
        y_norm: np.ndarray,
        x_target_norm: np.ndarray,
        scope: str,
    ) -> None:
        """Execute decision validation service if configured."""
        if self._dv_service is None:
            return

        ood_calibrator, conformal_calibrator = self._get_decision_validation_calibrators(
            scope
        )

        report = self._dv_service.validate(
            candidate=np.asarray(y_norm[0]),
            target=np.asarray(x_target_norm[0]),
            ood_calibrator=ood_calibrator,
            conformal_calibrator=conformal_calibrator,
        )
        passed = report.verdict is Verdict.ACCEPT
        if self._logger:
            summary = {
                "assurance_verdict_accept": 1.0 if passed else 0.0,
                "assurance_gate1_md2": report.metrics.get("gate1_md2"),
                "assurance_gate1_thr": report.metrics.get("gate1_md2_threshold"),
                "assurance_gate2_q": report.metrics.get("gate2_conformal_radius_q"),
                "assurance_gate2_dist": report.metrics.get("gate2_dist_to_target_l2"),
            }
            self._logger.log_metrics(summary)
            self._logger.log_info(
                f"[assurance] Verdict={report.verdict.value} | Reasons={report.explanations}"
            )

    def _get_decision_validation_calibrators(
        self, scope: str
    ) -> tuple[BaseOODCalibrator, BaseConformalCalibrator]:
        calibrators = self._dv_calibrators.get(scope)
        if calibrators is not None:
            return calibrators

        if self._dv_calibration_repository is None:
            raise RuntimeError(
                "Decision validation calibration repository is not configured."
            )

        calibration = self._dv_calibration_repository.load_latest(scope=scope)
        calibrators = (calibration.ood_calibrator, calibration.conformal_calibrator)
        self._dv_calibrators[scope] = calibrators

        self._log_loaded_calibration(scope, calibrators)
        return calibrators

    def _log_loaded_calibration(
        self,
        scope: str,
        calibrators: tuple[BaseOODCalibrator, BaseConformalCalibrator],
    ) -> None:
        if not self._logger:
            return

        ood_calibrator, conformal_calibrator = calibrators
        message_parts: list[str] = [f"scope={scope}"]

        try:
            threshold = float(ood_calibrator.threshold)
            message_parts.append(f"OOD thr={threshold:.3f}")
        except Exception:
            message_parts.append("OOD thr=<unavailable>")

        try:
            radius = float(conformal_calibrator.radius)
            message_parts.append(f"Conformal q={radius:.4f}")
        except Exception:
            message_parts.append("Conformal q=<unavailable>")

        self._logger.log_info("[assurance] Loaded " + ", ".join(message_parts))
