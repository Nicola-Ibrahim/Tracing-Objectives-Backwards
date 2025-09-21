from __future__ import annotations

import numpy as np

from ....feasibility.value_objects.tolerance import Tolerance
from ...entities.generated_decision_validation_report import (
    GeneratedDecisionValidationReport,
    Verdict,
)
from ...value_objects.validation_outcome import ValidationOutcome
from ...strategies.calibration import ConformalCalibration, OODCalibration


def _md2(x: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> float:
    x = np.atleast_2d(x)
    d = x - mu
    return float(np.einsum("ni,ij,nj->n", d, prec, d)[0])


def evaluate_two_gate_policy(
    *,
    x_norm: np.ndarray,
    y_star_norm: np.ndarray,
    y_hat_norm: np.ndarray,
    ood: OODCalibration,
    conf: ConformalCalibration,
    tol: Tolerance,
) -> GeneratedDecisionValidationReport:
    metrics: dict[str, float | bool] = {}
    explanations: dict[str, str] = {}

    md2 = _md2(x_norm, ood.mu, ood.prec)
    inlier = md2 <= ood.threshold_md2
    metrics.update(
        {
            "gate1_md2": md2,
            "gate1_md2_threshold": ood.threshold_md2,
            "gate1_inlier": bool(inlier),
        }
    )

    if not inlier:
        explanations["gate1"] = "ABSTAIN: decision is outside the supported region."
        return GeneratedDecisionValidationReport(
            verdict=Verdict.ABSTAIN,
            metrics=metrics,
            explanations=explanations,
        )

    explanations["gate1"] = "Pass: decision within supported region."

    diff = np.abs(y_hat_norm - y_star_norm)
    dist_l2 = float(np.linalg.norm(diff))
    q = conf.radius_q

    if tol.eps_l2 is not None:
        covered = (dist_l2 + q) <= tol.eps_l2
    elif tol.eps_per_obj is not None:
        covered = bool(np.all(diff + q <= tol.eps_per_obj))
    else:
        raise ValueError("Tolerance requires eps_l2 or eps_per_obj.")

    metrics.update(
        {
            "gate2_conformal_radius_q": q,
            "gate2_dist_to_target_l2": dist_l2,
            "gate2_covered": bool(covered),
        }
    )

    if covered:
        explanations["gate2"] = "Pass: predictive set contained within tolerance."
        verdict = Verdict.ACCEPT
    else:
        explanations["gate2"] = "ABSTAIN: uncertainty exceeds tolerance."
        verdict = Verdict.ABSTAIN

    return GeneratedDecisionValidationReport(
        verdict=verdict,
        metrics=metrics,
        explanations=explanations,
    )


__all__ = ["evaluate_two_gate_policy"]
