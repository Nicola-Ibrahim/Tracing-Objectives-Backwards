import numpy as np

from ....feasibility.value_objects.tolerance import Tolerance
from ...entities.generated_decision_validation_report import (
    GeneratedDecisionValidationReport,
)
from ...value_objects.calibration import ConformalCalibration, OODCalibration
from ...value_objects.gate_result import GateResult
from ...value_objects.validation_outcome import Verdict


def _md2(x: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> float:
    """Compute Mahalanobis distance squared for a single sample."""
    x = np.atleast_2d(x)
    d = x - mu
    return float(np.einsum("ni,ij,nj->n", d, prec, d)[0])


def evaluate_two_gate_policy(
    *,
    y_norm: np.ndarray,
    x_target_norm: np.ndarray,
    x_hat_norm: np.ndarray,
    ood: OODCalibration,
    conf: ConformalCalibration,
    tol: Tolerance,
) -> GeneratedDecisionValidationReport:
    """Apply OOD and tolerance gates to produce a validation report."""
    metrics: dict[str, float | bool] = {}
    explanations: dict[str, str] = {}
    gate_results: list[GateResult] = []

    md2 = _md2(y_norm, ood.mu, ood.prec)
    inlier = md2 <= ood.threshold_md2
    metrics.update(
        {
            "gate1_md2": md2,
            "gate1_md2_threshold": ood.threshold_md2,
            "gate1_inlier": bool(inlier),
        }
    )
    gate_results.append(
        GateResult(
            name="gate1",
            passed=bool(inlier),
            metrics={"md2": md2, "threshold": ood.threshold_md2},
            explanation=(
                "Pass: y within supported region."
                if inlier
                else "ABSTAIN: y is outside the supported region."
            ),
        )
    )

    if not inlier:
        explanations["gate1"] = gate_results[-1].explanation
        return GeneratedDecisionValidationReport(
            verdict=Verdict.ABSTAIN,
            metrics=metrics,
            explanations=explanations,
            gate_results=tuple(gate_results),
        )

    explanations["gate1"] = gate_results[-1].explanation

    diff = np.abs(x_hat_norm - x_target_norm)
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
    gate_results.append(
        GateResult(
            name="gate2",
            passed=bool(covered),
            metrics={"radius_q": q, "dist_to_target_l2": dist_l2},
            explanation=(
                "Pass: predicted x within tolerance."
                if covered
                else "ABSTAIN: x uncertainty exceeds tolerance."
            ),
        )
    )

    explanations["gate2"] = gate_results[-1].explanation
    verdict = Verdict.ACCEPT if covered else Verdict.ABSTAIN

    return GeneratedDecisionValidationReport(
        verdict=verdict,
        metrics=metrics,
        explanations=explanations,
        gate_results=tuple(gate_results),
    )


__all__ = ["evaluate_two_gate_policy"]
