import numpy as np

from ..entities.report import GeneratedDecisionValidationReport, Verdict
from ..value_objects import ConformalCalibration, OODCalibration, Tolerances


def _md2(x: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> float:
    x = np.atleast_2d(x)
    d = x - mu
    return float(np.einsum("ni,ij,nj->n", d, prec, d)[0])


def evaluate_two_gate_policy(
    *,
    x_norm: np.ndarray,  # (d_x,)
    y_star_norm: np.ndarray,  # (d_y,)
    y_hat_norm: np.ndarray,  # (d_y,)
    ood: OODCalibration,
    conf: ConformalCalibration,
    tol: Tolerances,
) -> GeneratedDecisionValidationReport:
    """
    Gate1: inlier check with MD^2 threshold on X.
    Gate2: split-conformal joint-L2 coverage within user tolerance around y*.
    """
    metrics: dict[str, float | int] = {}
    explanations: dict[str, str] = {}

    # Gate 1: OOD on X
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
        explanations["gate1"] = "ABSTAIN: decision lies outside supported region (OOD)."
        return GeneratedDecisionValidationReport(
            verdict=Verdict.ABSTAIN, metrics=metrics, explanations=explanations
        )
    explanations["gate1"] = "Pass: decision is in-distribution."

    # Gate 2: predictive set vs tolerance around y*
    q = conf.radius_q
    diff = np.abs(y_hat_norm - y_star_norm)
    dist_l2 = float(np.linalg.norm(y_hat_norm - y_star_norm))
    covered: bool
    if tol.eps_l2 is not None:
        covered = (dist_l2 + q) <= tol.eps_l2
    elif tol.eps_per_obj is not None:
        covered = bool(np.all(diff + q <= tol.eps_per_obj))
    else:
        raise ValueError("Provide either eps_l2 or eps_per_obj in Tolerances.")

    metrics.update(
        {
            "gate2_conformal_radius_q": q,
            "gate2_dist_to_target_l2": dist_l2,
            "gate2_tolerance_l2": float(tol.eps_l2) if tol.eps_l2 is not None else None,
            "gate2_covered": bool(covered),
        }
    )

    if covered:
        explanations["gate2"] = "Pass: predictive uncertainty fits within tolerance."
        verdict = Verdict.ACCEPT
    else:
        explanations["gate2"] = (
            "ABSTAIN: predictive uncertainty too large for tolerance."
        )
        verdict = Verdict.ABSTAIN

    return GeneratedDecisionValidationReport(
        verdict=verdict, metrics=metrics, explanations=explanations
    )
