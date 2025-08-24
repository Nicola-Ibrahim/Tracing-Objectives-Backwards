import numpy as np

from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.model_management.interfaces.base_logger import BaseLogger
from ....domain.model_management.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ....domain.objective.feasibility.checker import ObjectiveFeasibilityChecker
from ....domain.objective.feasibility.exceptions import ObjectiveOutOfBoundsError
from ....domain.objective.feasibility.scorers import (
    ConvexHullScoreStrategy,
    KDEScoreStrategy,
    LocalSphereScoreStrategy,
    MinDistanceScoreStrategy,
)
from ....infrastructure.problems.biobj import get_coco_problem
from .generate_decision_command import GenerateDecisionCommand

# TODO: Move and relocate the error calculations parts to different method or class
# TODO: Imporve the feasibility checker to be more flexible and accept more args from `GenerateDecisionCommand`
# TODO: Refine and enhance the info and error msgs


class GenerateDecisionCommandHandler:
    def __init__(
        self,
        interpolation_model_repo: BaseInterpolationModelRepository,
        data_repository: BaseParetoDataRepository,
        logger: BaseLogger,
    ):
        self._interpolation_model_repo = interpolation_model_repo
        self._data_repository = data_repository
        self._logger = logger

    def execute(self, command: GenerateDecisionCommand) -> np.ndarray:
        self._logger.log_info(
            f"Starting FreeMode decision generation for interpolator type: {command.model_type}"
        )

        # Load interpolation model and normalizers
        model = self._interpolation_model_repo.get_latest_version(
            model_type=command.model_type
        )
        self._logger.log_info(
            f"Loaded interpolation model version {model.version} "
            f"of type {command.model_type}."
        )

        ml_mapper = model.ml_mapper
        decisions_normalizer = model.decisions_normalizer
        objectives_normalizer = model.objectives_normalizer

        # Load Pareto data
        raw_data = self._data_repository.load("pareto_data")

        # Load original COCO problem for ground-truth objective evaluation
        # Assuming get_coco_problem returns a callable that takes a decision and returns objectives
        coco_problem = get_coco_problem(
            function_indices=5
        )  # Example: function 5 (two objectives)
        self._logger.log_info(
            f"Using COCO problem: func_id={coco_problem.id}, "
            f"instance={coco_problem.id_instance}, dim={coco_problem.dimension}"
        )

        # Normalize user input
        target_objective = np.array(command.target_objective)
        if target_objective.ndim == 1:
            target_objective = target_objective.reshape(
                1, -1
            )  # Ensure 2D for normalizer

        target_objective_normalized = objectives_normalizer.transform(target_objective)
        self._logger.log_info(
            f"Target objective: {target_objective[0]} (Normalized: {target_objective_normalized[0]})."
        )

        pareto_front_normalized = objectives_normalizer.transform(raw_data.pareto_front)

        try:
            # Validate that the target is feasible
            checker = ObjectiveFeasibilityChecker(
                pareto_front=raw_data.pareto_front,
                pareto_front_normalized=pareto_front_normalized,
                tolerance=command.distance_tolerance,
                scorer=KDEScoreStrategy(),
            )
            checker.validate(
                target=target_objective,
                target_normalized=target_objective_normalized,
                num_suggestions=command.num_suggestions,
            )
            self._logger.log_info("✅ Target objective passed feasibility check.")

            # Predict normalized decision
            decision_pred_norm = ml_mapper.predict(
                target_objective_normalized
            )
            self._logger.log_info(
                f"Normalized predicted decision: {decision_pred_norm}."
            )

            # Convert decision back to original space
            # inverse_transform usually returns 2D array, take first row for single prediction
            decision_pred = decisions_normalizer.inverse_transform(decision_pred_norm)[
                0
            ]
            self._logger.log_info(
                f"Inverse-transformed predicted decision: {decision_pred}."
            )

            # Evaluate the decision using the true COCO problem
            true_objective = np.array(coco_problem(decision_pred))

            self._logger.log_info(f"🎯 Target objective (F): {target_objective[0]}")
            self._logger.log_info(f"🎯 Achieved objective (F'): {true_objective}")

            # --- Improved Error Calculation ---
            # Ensure both are 1D arrays for element-wise operations if they come in as 2D (1, N)
            target_obj_1d = (
                target_objective[0] if target_objective.ndim == 2 else target_objective
            )

            # 1. Absolute Difference (per objective)
            abs_diff = np.abs(true_objective - target_obj_1d)
            formatted_abs_diff = [f"{v:.6f}" for v in abs_diff]
            self._logger.log_info(
                f"📏 Absolute error per objective: [{', '.join(formatted_abs_diff)}]"
            )

            # 2. Euclidean Distance in Objective Space
            euclidean_distance = np.linalg.norm(true_objective - target_obj_1d)
            self._logger.log_info(
                f"📏 Euclidean distance in objective space (L2-norm): {euclidean_distance:.6f}"
            )

            # 3. Maximum Absolute Error (L-infinity norm)
            max_abs_error = np.max(abs_diff)
            self._logger.log_info(
                f"📏 Maximum absolute error across objectives (L_inf-norm): {max_abs_error:.6f}"
            )

            # 4. Mean Absolute Error (MAE)
            mean_abs_error = np.mean(abs_diff)
            self._logger.log_info(
                f"📏 Mean absolute error across objectives (MAE): {mean_abs_error:.6f}"
            )

            # 5. Relative Difference (per objective) - kept for individual objective insight
            # Using np.maximum to prevent division by zero or very small numbers
            rel_diff = abs_diff / np.maximum(np.abs(target_obj_1d), 1e-8)
            formatted_rel_diff = [f"{v:.2%}" for v in rel_diff]  # Format as percentage
            self._logger.log_info(
                f"📐 Relative error per objective (compared to target): [{', '.join(formatted_rel_diff)}]"
            )

            # Optional: Add warning based on aggregated metric (e.g., Euclidean distance or Max Abs Error)
            # Thresholds might need to be fine-tuned based on problem scale and acceptable deviation.
            # A good practice might be to normalize the error by the typical range of the objectives
            # for a more robust threshold. For this example, let's use a fixed value.

            # Warning based on Euclidean distance
            # (Example threshold, adjust based on expected objective scales)
            if euclidean_distance > 0.1:
                self._logger.log_warning(
                    f"⚠️ Achieved objective's Euclidean distance from target is significant ({euclidean_distance:.6f})."
                )

            # Warning based on maximum relative deviation across any single objective
            if np.any(rel_diff > 0.1):  # If any objective deviates by more than 10%
                self._logger.log_warning(
                    f"⚠️ One or more objectives show significant relative deviation (>10%)."
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
            if e.extra_info:
                self._logger.log_error(f"   Details: {e.extra_info}")

            if e.suggestions is not None and len(e.suggestions) > 0:
                possibilities = []
                for s_norm in e.suggestions:
                    s_unnorm = objectives_normalizer.inverse_transform(
                        s_norm.reshape(1, -1)
                    )[0]
                    # Dynamically format suggestions for N objectives
                    s_str = ", ".join(
                        [f"f{i + 1}: {val:.4f}" for i, val in enumerate(s_unnorm)]
                    )
                    possibilities.append(f"\n   - {s_str}")

                self._logger.log_error(
                    "   Nearby feasible objectives you can try (original scale):"
                    f"{''.join(possibilities)}"
                )
