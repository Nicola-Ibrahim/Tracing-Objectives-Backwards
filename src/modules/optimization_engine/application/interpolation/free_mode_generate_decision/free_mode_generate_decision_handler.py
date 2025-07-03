import numpy as np

from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.interpolation.exceptions import ObjectiveOutOfBoundsError
from ....domain.interpolation.interfaces.base_logger import BaseLogger
from ....domain.interpolation.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ....domain.services.feasibility_checker import ObjectiveFeasibilityChecker
from ....infrastructure.problems.biobj import get_coco_problem
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

        # Load Pareto data
        raw_data = self._paret_data_repo.load("pareto_data")

        # Load original COCO problem for ground-truth objective evaluation
        coco_problem = get_coco_problem(function_indices=5)
        self._logger.log_info(
            f"Using COCO problem: func_id={coco_problem.id}, "
            f"instance={coco_problem.id_instance}, dim={coco_problem.dimension}"
        )

        # Normalize user input
        target_objective = np.array(command.target_objective)
        if target_objective.ndim == 1:
            target_objective = target_objective.reshape(1, -1)

        target_objective_norm = objectives_normalizer.transform(target_objective)
        self._logger.log_info(
            f"Target objective: {target_objective}. Normalized: {target_objective_norm}."
        )

        pareto_front_norm = objectives_normalizer.transform(raw_data.pareto_front)

        try:
            # Validate that the target is feasible
            checker = ObjectiveFeasibilityChecker(
                pareto_front=raw_data.pareto_front,
                pareto_front_norm=pareto_front_norm,
                tolerance=command.distance_tolerance,
            )
            checker.validate(
                target=target_objective,
                target_norm=target_objective_norm,
                num_suggestions=command.num_suggestions,
            )
            self._logger.log_info("✅ Target objective passed feasibility check.")

            # Predict normalized decision
            decision_pred_norm = inverse_decision_mapper.predict(target_objective_norm)
            self._logger.log_info(
                f"Normalized predicted decision: {decision_pred_norm}."
            )

            # Convert decision back to original space
            decision_pred = decisions_normalizer.inverse_transform(decision_pred_norm)[
                0
            ]
            self._logger.log_info(
                f"Inverse-transformed predicted decision: {decision_pred}."
            )

            # ✅ Evaluate the decision using the true COCO problem
            true_objective = np.array(coco_problem(decision_pred))
            abs_diff = np.abs(true_objective - target_objective[0])
            rel_diff = abs_diff / np.maximum(np.abs(target_objective[0]), 1e-8)

            self._logger.log_info(f"🎯 Evaluated objective from COCO: {true_objective}")
            self._logger.log_info(f"🎯 Target objective: {target_objective[0]}")
            self._logger.log_info(f"📏 Absolute error: {abs_diff}")
            formatted_rel_diff = [f"{v:.6f}" for v in rel_diff]
            self._logger.log_info(
                f"📐 Relative error: [{', '.join(formatted_rel_diff)}]"
            )

            # Optional: Add threshold for deviation warning
            if np.any(rel_diff > 0.1):  # 10% relative deviation
                self._logger.log_warning(
                    f"⚠️ Evaluated objective deviates significantly (>10%) from the target."
                )

            self._logger.log_info(
                "🎉 FreeMode decision generation completed successfully."
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
                    "   Nearby feasible objectives you can try (original scale):"
                )
                for s_norm in e.suggestions:
                    s_unnorm = objectives_normalizer.inverse_transform(
                        s_norm.reshape(1, -1)
                    )[0]
                    self._logger.log_error(
                        f"   - f1: {s_unnorm[0]:.4f}, f2: {s_unnorm[1]:.4f}"
                    )
