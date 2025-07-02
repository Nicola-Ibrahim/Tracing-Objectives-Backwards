import numpy as np

from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.interpolation.exceptions import ObjectiveOutOfBoundsError
from ....domain.interpolation.interfaces.base_logger import BaseLogger
from ....domain.interpolation.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ....domain.services.feasibility_checker import ObjectiveFeasibilityChecker
from .free_mode_generate_decision_command import FreeModeGenerateDecisionCommand


class FreeModeGenerateDecisionCommandHandler:
    def __init__(
        self,
        interpolation_model_repo: BaseInterpolationModelRepository,
        pareto_data_repo: BaseParetoDataRepository,
        logger: BaseLogger,
    ):
        self._interpolation_model_repo = interpolation_model_repo
        self._paret_data_repo = pareto_data_repo
        self._logger = logger

    def execute(self, command: FreeModeGenerateDecisionCommand) -> np.ndarray:
        self._logger.log_info(
            f"Starting FreeMode decision generation for interpolator type: {command.interpolator_type}"
        )

        # Load interpolation model and normalizers
        model = self._interpolation_model_repo.get_latest_version(
            interpolator_type=command.interpolator_type
        )
        self._logger.log_info(
            f"Loaded interpolation model version {model.version_number} "
            f"of type {command.interpolator_type}."
        )

        inverse_decision_mapper = model.inverse_decision_mapper
        decisions_normalizer = model.decisions_normalizer
        objectives_normalizer = model.objectives_normalizer

        # Load and normalize Pareto front
        raw_data = self._paret_data_repo.load("pareto_data")

        target_objective_array = np.array(command.target_objective)
        # Ensure target_objective_array is 2D for normalizer.transform if it expects it
        if target_objective_array.ndim == 1:
            target_objective_array = target_objective_array.reshape(1, -1)

        objective_norm = objectives_normalizer.transform(target_objective_array)
        self._logger.log_info(
            f"Target objective: {target_objective_array}. Normalized: {objective_norm}."
        )

        normalized_front = objectives_normalizer.transform(raw_data.pareto_front)

        try:
            objective_feasibility_checker = ObjectiveFeasibilityChecker(
                normalized_pareto_front=normalized_front,
                tolerance=command.distance_tolerance,
            )

            objective_feasibility_checker.validate(objective_norm)

        except ObjectiveOutOfBoundsError as e:
            self._logger.log_error(
                "❌ The selected objective is outside the feasible Pareto front."
            )
            self._logger.log_error(f"Closest distance: {e.distance:.4f}")
            self._logger.log_error(
                "Here are some nearby feasible objectives you can try:"
            )
            self._logger.log_error(f"Feasibility check failed: {e}")

            for s in objective_feasibility_checker.get_nearest_suggestions(
                target_objective_array, command.num_suggestions
            ):
                rounded = tuple(round(x, 4) for x in s)
                rounded = objectives_normalizer.inverse_transform(
                    np.array(rounded).reshape(1, -1)
                )
                self._logger.log_error(f"f1: {rounded[0][0]}, f2: {rounded[0][1]}")
        else:
            # Proceed with interpolation
            self._logger.log_info(
                f"Predicting decision for normalized objective: {objective_norm}."
            )
            decision_pred_norm = inverse_decision_mapper.predict(objective_norm)
            self._logger.log_info(
                f"Normalized predicted decision: {decision_pred_norm}."
            )

            # Inverse-transform predictions to original scale.
            # Note: predict typically returns 2D array, so [0] is used to get the 1D result for a single prediction.
            decision_pred = decisions_normalizer.inverse_transform(decision_pred_norm)[
                0
            ]
            self._logger.log_info(
                f"Inverse-transformed predicted decision: {decision_pred}."
            )

            return decision_pred
