import numpy as np

from modules.evaluation.domain.decision_validation.interfaces.base_conformal_calibrator import (
    BaseConformalValidator,
)
from modules.evaluation.domain.decision_validation.interfaces.base_ood_calibrator import (
    BaseOODValidator,
)


class CandidateValidator:
    """Validates candidates using optional OOD and Conformal validators."""

    def __init__(
        self,
        ood_validator: BaseOODValidator | None = None,
        conformal_validator: BaseConformalValidator | None = None,
    ) -> None:
        self._ood_validator = ood_validator
        self._conformal_validator = conformal_validator

    @property
    def is_enabled(self) -> bool:
        """Returns True if at least one validator is configured."""
        return self._ood_validator is not None or self._conformal_validator is not None

    def validate(
        self,
        candidates_raw: np.ndarray,
        predicted_objectives: np.ndarray,
        target_objective_raw: np.ndarray,
        distance_tolerance: float,
    ) -> np.ndarray:
        """
        Returns a boolean mask of valid candidates.
        """
        if not self.is_enabled:
            return np.ones(candidates_raw.shape[0], dtype=bool)

        candidates_raw = np.asarray(candidates_raw, dtype=float)
        target_flat = np.asarray(target_objective_raw, dtype=float).reshape(-1)
        predicted_objectives = np.asarray(predicted_objectives, dtype=float)

        mask = np.ones(candidates_raw.shape[0], dtype=bool)

        # Gate 1: OOD validation on decision candidates.
        if self._ood_validator is not None:
            mask &= self._validate_ood(self._ood_validator, candidates_raw)

        # Gate 2: conformal validation on predicted objectives for the remaining candidates.
        if self._conformal_validator is not None and mask.any():
            objectives_inlier = predicted_objectives[mask]
            conformal_mask = self._validate_conformal(
                self._conformal_validator,
                objectives_inlier,
                target_flat,
                distance_tolerance,
            )
            mask_indices = np.where(mask)[0]
            mask[mask_indices] &= conformal_mask

        return mask

    @staticmethod
    def _validate_ood(ood_validator: BaseOODValidator, X: np.ndarray) -> np.ndarray:
        passed_flags: list[bool] = []
        for row in X:
            passed, _, _ = ood_validator.validate(row)
            passed_flags.append(bool(passed))
        return np.asarray(passed_flags, dtype=bool)

    @staticmethod
    def _validate_conformal(
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
