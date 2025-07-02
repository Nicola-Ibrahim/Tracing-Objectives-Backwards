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

        # Load raw Pareto data
        raw_data = self._paret_data_repo.load("pareto_data")

        target_objective = np.array(command.target_objective)
        # Ensure target_objective is 2D for normalizer.transform if it expects it
        if target_objective.ndim == 1:
            target_objective = target_objective.reshape(1, -1)

        target_objective_norm = objectives_normalizer.transform(target_objective)
        self._logger.log_info(
            f"Target objective: {target_objective}. Normalized: {target_objective_norm}."
        )

        pareto_front_norm = objectives_normalizer.transform(raw_data.pareto_front)

        try:
            # Initialize feasibility checker with both unnormalized and normalized fronts
            objective_feasibility_checker = ObjectiveFeasibilityChecker(
                pareto_front=raw_data.pareto_front,
                pareto_front_norm=pareto_front_norm,
                tolerance=command.distance_tolerance,
            )
            self._logger.log_info(
                f"Initiating feasibility check for target objective with tolerance {command.distance_tolerance}."
            )
            # Validate, passing both unnormalized and normalized targets
            objective_feasibility_checker.validate(
                target_objective,
                target_objective_norm,
                num_suggestions=command.num_suggestions,
            )
            self._logger.log_info(
                "Feasibility check completed successfully. Objective is feasible."
            )

            # Proceed with interpolation if validation passes
            self._logger.log_info(
                f"Predicting decision for normalized objective: {target_objective_norm}."
            )
            decision_pred_norm = inverse_decision_mapper.predict(target_objective_norm)
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

            self._logger.log_info(
                "FreeMode decision generation completed successfully."
            )
            return decision_pred

        except ObjectiveOutOfBoundsError as e:
            self._logger.log_error(
                f"❌ Feasibility check failed. Reason: {e.reason.value}"
            )
            self._logger.log_error(f"   Message: {e.message}")
            if e.distance is not None:
                self._logger.log_error(
                    f"   Closest distance to front: {e.distance:.4f}"
                )
            if e.extra_info:
                self._logger.log_error(f"   Details: {e.extra_info}")

            if e.suggestions is not None and len(e.suggestions) > 0:
                self._logger.log_error(
                    "   Here are some nearby feasible objectives you can try (original scale):"
                )
                # Inverse-transform suggestions for logging in original scale
                for s_norm in e.suggestions:
                    # Ensure s_norm is 2D for inverse_transform
                    s_unnorm = objectives_normalizer.inverse_transform(
                        s_norm.reshape(1, -1)
                    )[0]
                    # Assuming 2 objectives (f1, f2)
                    self._logger.log_error(
                        f"   - f1: {s_unnorm[0]:.4f}, f2: {s_unnorm[1]:.4f}"
                    )
