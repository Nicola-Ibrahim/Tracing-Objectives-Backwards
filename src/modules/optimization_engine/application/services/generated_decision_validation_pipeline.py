from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...domain.assurance.entities.generated_decision_validation_report import (
    GeneratedDecisionValidationReport,
)
from ...domain.assurance.services.validate.calibration import (
    calibrate_mahalanobis,
    calibrate_split_conformal_l2,
)
from ...domain.assurance.services.validate.policy import evaluate_two_gate_policy
from ...domain.assurance.value_objects import (
    ConformalCalibration,
    OODCalibration,
    Tolerances,
)
from ...domain.common.interfaces.base_logger import BaseLogger
from ...domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ...domain.modeling.interfaces.base_estimator import BaseEstimator
from ...domain.modeling.services.forward_ensemble import ForwardEnsemble


@dataclass
class DecisionValidationConfig:
    confidence: float = 0.90  # split-conformal target coverage (normalised Y)
    ood_percentile: float = 97.5  # MD^2 empirical threshold on normalised X
    y_tolerance_l2: float | None = 0.03
    y_tolerance_per_obj: np.ndarray | None = None
    cov_reg: float = 1e-6


class ValidateGeneratedDecisionService:
    """
    Thin application service:
      - loads calibration splits & normalisers
      - calibrates once (immutables)
      - predicts y_hat via ensemble
      - calls pure domain policy to produce a report
    """

    def __init__(
        self,
        processed_repo: BaseDatasetRepository,
        forward_estimators: Sequence[BaseEstimator],
        config: DecisionValidationConfig,
        logger: BaseLogger | None = None,
    ) -> None:
        self._processed_repo = processed_repo
        self._ensemble = ForwardEnsemble(forward_estimators)
        self._cfg = config
        self._logger = logger
        self._ood: OODCalibration | None = None
        self._conf: ConformalCalibration | None = None

    def _ensure_calibrated(self) -> None:
        if self._ood is not None and self._conf is not None:
            return
        # Expect 'calibration' artifact; fallback to 'dataset' train split
        processed = None
        Xn, Yn = None, None
        try:
            processed = self._processed_repo.load("calibration")
            Xn = processed.X_normalizer.transform(processed.X_cal)
            Yn = processed.y_normalizer.transform(processed.y_cal)
        except Exception:
            processed = self._processed_repo.load("dataset")
            Xn = processed.X_normalizer.transform(processed.X_train)
            Yn = processed.y_normalizer.transform(processed.y_train)

        # Domain calibrations (immutable) — pure functions
        self._ood = calibrate_mahalanobis(
            Xn, percentile=self._cfg.ood_percentile, cov_reg=self._cfg.cov_reg
        )
        self._conf = calibrate_split_conformal_l2(
            Xn, Yn, self._ensemble, confidence=self._cfg.confidence
        )

        if self._logger:
            self._logger.log_info(
                f"[assurance] Calibrated OOD thr={self._ood.threshold_md2:.3f}, "
                f"Conformal q={self._conf.radius_q:.4f}"
            )

    def validate(
        self, x_original: np.ndarray, y_star_original: np.ndarray
    ) -> GeneratedDecisionValidationReport:
        self._ensure_calibrated()

        processed = self._processed_repo.load("dataset")
        x_norm = processed.X_normalizer.transform(np.atleast_2d(x_original))[0]
        y_star_norm = processed.y_normalizer.transform(np.atleast_2d(y_star_original))[
            0
        ]

        # Predict ensemble mean in NORMALISED space
        y_hat_norm = self._ensemble.predict_mean(np.atleast_2d(x_norm))[0]

        tol = Tolerances(
            eps_l2=self._cfg.y_tolerance_l2, eps_per_obj=self._cfg.y_tolerance_per_obj
        )
        report = evaluate_two_gate_policy(
            x_norm=x_norm,
            y_star_norm=y_star_norm,
            y_hat_norm=y_hat_norm,
            ood=self._ood,  # type: ignore[arg-type]
            conf=self._conf,  # type: ignore[arg-type]
            tol=tol,
        )

        if self._logger:
            summary = {
                "assurance_verdict_accept": 1.0
                if report.verdict.value == "ACCEPT"
                else 0.0,
                "assurance_gate1_md2": report.metrics.get("gate1_md2"),
                "assurance_gate1_thr": report.metrics.get("gate1_md2_threshold"),
                "assurance_gate2_q": report.metrics.get("gate2_conformal_radius_q"),
                "assurance_gate2_dist": report.metrics.get("gate2_dist_to_target_l2"),
            }
            self._logger.log_metrics(summary)
            self._logger.log_info(
                f"[assurance] Verdict={report.verdict.value} | Reasons={report.explanations}"
            )

        return report
