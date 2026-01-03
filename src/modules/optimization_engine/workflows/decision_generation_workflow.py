import numpy as np

from ..domain.assurance.decision_validation.interfaces.base_conformal_calibrator import (
    BaseConformalValidator,
)
from ..domain.assurance.decision_validation.interfaces.base_ood_calibrator import (
    BaseOODValidator,
)
from ..domain.modeling.interfaces.base_estimator import BaseEstimator
from ..domain.modeling.interfaces.base_normalizer import BaseNormalizer


class DecisionGenerationWorkflow:
    """
    Three-station workflow for decision generation:
    1) Modeling   : sample candidate decisions (X) from inverse estimators given a target objective (y*).
    2) Verification: run the forward estimator to obtain predicted objectives y_hat = f(X) for each candidate.
    3) Assurance  : optionally validate/prune candidates using OOD + conformal validators.
    4) Selection  : pick the best candidate per estimator by distance to the target objective.
    """

    def __init__(
        self,
        conformal_validator: BaseConformalValidator | None = None,
        ood_validator: BaseOODValidator | None = None,
    ) -> None:
        self._conformal_validator = conformal_validator
        self._ood_validator = ood_validator

    def run(
        self,
        *,
        inverse_estimators: list[tuple[str, BaseEstimator]],
        pareto_front: np.ndarray,
        target_objective_raw: np.ndarray,
        target_objective_norm: np.ndarray,
        decisions_normalizer: BaseNormalizer,
        forward_estimator: BaseEstimator | None = None,
        n_samples: int,
        distance_tolerance: float,
    ) -> dict[str, object]:
        """Generate, (optionally) validate, and select decisions per inverse estimator.

        Returns a plain dictionary to keep the workflow lightweight and easy to integrate:
          - results_map: per-estimator results including best candidate and distances
          - generator_runs: pre-validation generation runs for visualization/debugging
        """
        results_map: dict[str, dict] = {}
        generator_runs: list[dict[str, object]] = []

        if forward_estimator is None:
            raise ValueError("Provide 'forward_estimator' for decision generation.")

        # Run the pipeline independently for each inverse estimator so we can compare
        # how different generators sample the decision space for the same target.
        for display_name, estimator in inverse_estimators:
            # Station 1 (Modeling): sample candidate decisions in *raw* decision space.
            # Inverse estimators typically operate in normalized objective space and output
            # normalized decisions; we denormalize back to raw decisions for the forward model.
            candidates_raw = self._sample_candidates(
                estimator=estimator,
                target_objective_norm=target_objective_norm,
                n_samples=n_samples,
                decisions_normalizer=decisions_normalizer,
            )

            # Station 2 (Verification): predict objectives for each candidate with the forward model.
            predicted_objectives = self._predict_outcomes(
                forward_estimator=forward_estimator,
                candidates_raw=candidates_raw,
            )

            # Keep the *pre-validation* run for visualization so we can inspect how each
            # inverse estimator samples, regardless of what later validation prunes away.
            generator_runs.append(
                {
                    "name": display_name,
                    "decisions": candidates_raw,
                    "predicted_objectives": predicted_objectives,
                }
            )

            # Station 3 (Assurance): optionally validate/prune candidates.
            # If no validators are configured, we skip this step entirely.
            has_validators = (
                self._ood_validator is not None or self._conformal_validator is not None
            )
            if has_validators:
                mask = self._apply_validation(
                    candidates_raw=candidates_raw,
                    predicted_objectives=predicted_objectives,
                    target_objective_raw=target_objective_raw,
                    distance_tolerance=distance_tolerance,
                )

                candidates_raw = candidates_raw[mask]
                predicted_objectives = predicted_objectives[mask]

            # Station 4 (Selection): choose the closest predicted objective to the target.
            # (This assumes "best" means minimum L2 distance in objective space.)
            best_idx, distances = self._select_best(
                predicted_objectives=predicted_objectives,
                target_objective_raw=target_objective_raw,
            )

            # Persist per-estimator outputs in a uniform dict shape for downstream use.
            results_map[display_name] = {
                "decisions": candidates_raw,
                "predicted_objectives": predicted_objectives,
                "best_index": int(best_idx),
                "best_distance": float(distances[best_idx]),
                "best_decision": candidates_raw[best_idx],
                "best_objective": predicted_objectives[best_idx],
            }

        return {
            "results_map": results_map,
            "generator_runs": generator_runs,
        }

    def _apply_validation(
        self,
        *,
        candidates_raw: np.ndarray,
        predicted_objectives: np.ndarray,
        target_objective_raw: np.ndarray,
        distance_tolerance: float,
    ) -> np.ndarray | None:
        """
        Validate candidates using configured validators (OOD + conformal).
        """
        # Defensive guard: if no validators are configured, the caller should skip,
        # but we also handle it here for safety.
        if self._ood_validator is None and self._conformal_validator is None:
            return None

        candidates_raw = np.asarray(candidates_raw, dtype=float)
        if candidates_raw.ndim != 2:
            raise ValueError("Expected candidates_raw to be 2-D (n_candidates, x_dim).")

        # Validators operate in different spaces:
        # - OOD validator: validates raw decisions X (inlier/outlier in decision space)
        # - Conformal validator: validates predicted objectives y_hat vs target y*
        target_flat = np.asarray(target_objective_raw, dtype=float).reshape(-1)
        predicted_objectives = np.asarray(predicted_objectives, dtype=float)

        mask = np.ones(candidates_raw.shape[0], dtype=bool)

        # Gate 1: OOD validation on decision candidates.
        if self._ood_validator is not None:
            mask &= self._validate_ood(
                ood_validator=self._ood_validator, X=candidates_raw
            )

        # Gate 2: conformal validation on predicted objectives for the remaining candidates.
        if self._conformal_validator is not None and mask.any():
            objectives_inlier = predicted_objectives[mask]
            conformal_mask = self._validate_conformal(
                validator=self._conformal_validator,
                y_pred=objectives_inlier,
                y_target=target_flat,
                tolerance=float(distance_tolerance),
            )
            mask_indices = np.where(mask)[0]
            mask[mask_indices] &= conformal_mask

        return mask

    @staticmethod
    def _validate_ood(*, ood_validator: BaseOODValidator, X: np.ndarray) -> np.ndarray:
        passed_flags: list[bool] = []
        for row in X:
            passed, _, _ = ood_validator.validate(row)
            passed_flags.append(bool(passed))
        return np.asarray(passed_flags, dtype=bool)

    @staticmethod
    def _validate_conformal(
        *,
        validator: BaseConformalValidator,
        y_pred: np.ndarray,
        y_target: np.ndarray,
        tolerance: float,
    ) -> np.ndarray:
        passed_flags: list[bool] = []
        for row in y_pred:
            passed, _, _ = validator.validate(
                y_pred=row, y_target=y_target, tolerance=float(tolerance)
            )
            passed_flags.append(bool(passed))
        return np.asarray(passed_flags, dtype=bool)

    @staticmethod
    def _select_best(
        *, predicted_objectives: np.ndarray, target_objective_raw: np.ndarray
    ) -> tuple[int, np.ndarray]:
        distances = np.linalg.norm(
            predicted_objectives - target_objective_raw.reshape(1, -1), axis=1
        )
        best_idx = int(np.argmin(distances))
        return best_idx, distances

    @staticmethod
    def _sample_candidates(
        *,
        estimator: BaseEstimator,
        target_objective_norm: np.ndarray,
        n_samples: int,
        decisions_normalizer: BaseNormalizer,
    ) -> np.ndarray:
        # Prefer probabilistic sampling if available. Deterministic estimators fall back
        # to predict() and we repeat the single output to match n_samples.
        if hasattr(estimator, "sample"):
            candidates_norm = estimator.sample(
                target_objective_norm, n_samples=n_samples
            )
        else:
            candidates_norm = estimator.predict(target_objective_norm)
            candidates_norm = np.asarray(candidates_norm, dtype=float)
            if candidates_norm.ndim == 1:
                candidates_norm = candidates_norm.reshape(1, -1)
            if n_samples > 1:
                candidates_norm = np.repeat(candidates_norm, repeats=n_samples, axis=0)

        # Normalize shapes to 2D: (n_candidates, x_dim).
        if candidates_norm.ndim == 3:
            candidates_norm = candidates_norm.reshape(-1, candidates_norm.shape[-1])

        # Convert normalized decision space back into raw (physical) decision space.
        candidates_raw = decisions_normalizer.inverse_transform(candidates_norm)
        return candidates_raw

    @staticmethod
    def _predict_outcomes(
        *, forward_estimator: BaseEstimator, candidates_raw: np.ndarray
    ) -> np.ndarray:
        # Forward model typically expects raw decisions and returns raw objectives.
        predictions = forward_estimator.predict(candidates_raw)
        predictions = np.asarray(predictions, dtype=float)

        # Normalize shapes to 2D: (n_candidates, y_dim).
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        elif predictions.ndim == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])

        return predictions
