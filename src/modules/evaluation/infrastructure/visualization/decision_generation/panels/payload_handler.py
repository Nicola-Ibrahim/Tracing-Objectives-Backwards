import numpy as np


def prepare_payload(data: object) -> dict:
    """Coerce and validate incoming data payload."""
    payload = _coerce_payload(data)

    # Validate Pareto Front
    pareto_front = np.asarray(payload["pareto_front"], dtype=float)
    if pareto_front.ndim == 1:
        pareto_front = pareto_front.reshape(-1, 1)
    if pareto_front.shape[1] < 2:
        raise ValueError("Pareto front must have at least two objective columns.")
    payload["pareto_front"] = pareto_front

    # Validate Pareto Set
    if "pareto_set" in payload:
        pareto_set = np.asarray(payload["pareto_set"], dtype=float)
        if pareto_set.ndim == 1:
            pareto_set = pareto_set.reshape(-1, 1)
        payload["pareto_set"] = pareto_set

    # Validate Target
    target = np.asarray(payload["target_objective"], dtype=float).reshape(-1)
    if target.size < 2:
        raise ValueError("Target objective must have at least two dimensions.")
    payload["target_objective"] = target

    return payload


def _coerce_payload(data: object) -> dict[str, object]:
    """Convert input data into a standardized dictionary format."""
    if isinstance(data, dict):
        return data

    # Compatibility for legacy object data
    if all(
        hasattr(data, attr)
        for attr in ["pareto_front", "target_objective", "generators"]
    ):
        return {
            "pareto_front": getattr(data, "pareto_front"),
            "pareto_set": getattr(data, "pareto_set", None),
            "target_objective": getattr(data, "target_objective"),
            "generators": [
                {
                    "name": getattr(run, "name", None),
                    "decisions": getattr(run, "decisions", None),
                    "predicted_objectives": getattr(run, "predicted_objectives", None),
                    "best_index": getattr(run, "best_index", None),
                    "best_decision": getattr(run, "best_decision", None),
                    "best_objective": getattr(run, "best_objective", None),
                }
                for run in getattr(data, "generators")
            ],
        }

    raise TypeError(
        "DecisionGenerationComparisonVisualizer expects a dict or object "
        "with 'pareto_front', 'target_objective', and 'generators'."
    )
