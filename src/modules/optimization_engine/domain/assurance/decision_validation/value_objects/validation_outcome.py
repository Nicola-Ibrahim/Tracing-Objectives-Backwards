from dataclasses import dataclass

from ..entities.generated_decision_validation_report import Verdict


@dataclass(slots=True, frozen=True)
class ValidationOutcome:
    verdict: Verdict
    gate_results: tuple[str, ...] = ()


__all__ = ["ValidationOutcome"]
