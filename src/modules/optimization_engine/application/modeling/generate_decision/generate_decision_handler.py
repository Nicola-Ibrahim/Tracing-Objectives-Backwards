import numpy as np

from ....domain.assurance.services.feasibility.exceptions import (
    ObjectiveOutOfBoundsError,
)
from ....domain.assurance.services.feasibility.scorers import KDEScoreStrategy
from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_estimator import BaseEstimator
from ....domain.modeling.interfaces.base_repository import (
    BaseModelArtifactRepository,
)
from ...services.objective_feasibility_checker import (
    ObjectiveFeasibilityChecker,
)
from .generate_decision_command import GenerateDecisionCommand


class GenerateDecisionCommandHandler:
    def __init__(
        self,
        model_repository: BaseModelArtifactRepository,
        processed_data_repository: BaseDatasetRepository,
        logger: BaseLogger,
        forward_model: BaseEstimator,
    ):
        self._model_repository = model_repository
        self._processed_repo = processed_data_repository
        self._logger = logger
        self._forward_model = forward_model

    def execute(self, command: GenerateDecisionCommand) -> np.ndarray:
        # Load estimator (latest version by type)
        artifact = self._model_repository.get_latest_version(
            estimator_type=command.estimator_type
        )
        estimator = artifact.estimator

        # Load normalizers and Pareto data from processed dataset
        processed = self._processed_repo.load("dataset")
        decisions_normalizer = processed.X_normalizer
        objectives_normalizer = processed.y_normalizer
        pareto_set = getattr(processed, "pareto_set", None)
        pareto_front = getattr(processed, "pareto_front", None)

        # Normalize target objective
        target_objective = np.array(command.target_objective)
        if target_objective.ndim == 1:
            target_objective = target_objective.reshape(1, -1)
        target_objective_norm = objectives_normalizer.transform(target_objective)

        try:
            # Feasibility check (if pareto_front available)
            if pareto_front is not None:
                pareto_front_norm = objectives_normalizer.transform(pareto_front)
                ObjectiveFeasibilityChecker(
                    pareto_front=pareto_front,
                    pareto_front_normalized=pareto_front_norm,
                    tolerance=command.distance_tolerance,
                    scorer=KDEScoreStrategy(),
                ).validate(
                    target=target_objective,
                    target_normalized=target_objective_norm,
                    num_suggestions=command.num_suggestions,
                )

            # Predict decision in normalized space and inverse-transform
            decision_pred_norm = estimator.predict(target_objective_norm)
            decision_pred = decisions_normalizer.inverse_transform(decision_pred_norm)[
                0
            ]

            # Evaluate achieved objective using injected forward estimator (forward mapper)
            achieved_objective = self._forward_model.predict(
                decision_pred.reshape(1, -1)
            )[0]

            # Compute concise metrics
            target_obj_1d = (
                target_objective[0] if target_objective.ndim == 2 else target_objective
            )
            abs_diff = np.abs(achieved_objective - target_obj_1d)
            euclidean_distance = float(
                np.linalg.norm(achieved_objective - target_obj_1d)
            )
            # Optional: distance to nearest Pareto decision (if available)
            decision_to_pareto_min_distance = None
            if pareto_set is not None:
                try:
                    dists = np.linalg.norm(pareto_set - decision_pred, axis=1)
                    decision_to_pareto_min_distance = float(np.min(dists))
                except Exception:
                    decision_to_pareto_min_distance = None

            # Single concise result log
            metrics: dict[str, float] = {
                "objective_l2_distance": euclidean_distance,
                "objective_abs_error_mean": float(np.mean(abs_diff)),
                "objective_abs_error_max": float(np.max(abs_diff)),
            }
            if decision_to_pareto_min_distance is not None:
                metrics["decision_to_pareto_min_distance"] = (
                    decision_to_pareto_min_distance
                )
            self._logger.log_metrics(metrics)
            self._logger.log_info(
                f"Decision: {decision_pred.tolist()}, AchievedObjective: {achieved_objective.tolist()}"
            )

            return decision_pred

        except ObjectiveOutOfBoundsError as e:
            self._logger.log_error(
                f"Feasibility failed: {e.reason.value} | {e.message}"
            )
            # If suggestions available, log minimally in original scale
            try:
                suggestions = (
                    np.asarray(e.suggestions) if e.suggestions is not None else None
                )
                if suggestions is not None and suggestions.size > 0:
                    suggestions = objectives_normalizer.inverse_transform(suggestions)
                    self._logger.log_error(
                        "Suggestions (original scale): "
                        + "; ".join(str(s.tolist()) for s in suggestions)
                    )
            except Exception:
                pass
            raise
