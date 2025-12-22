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
            "target_objective": getattr(data, "target_objective"),
            "generators": [
                {
                    "name": getattr(run, "name", None),
                    "decisions": getattr(run, "decisions", None),
                    "predicted_objectives": getattr(run, "predicted_objectives", None),
                }
                for run in getattr(data, "generators")
            ],
        }

    raise TypeError(
        "DecisionGenerationComparisonVisualizer expects a dict or object "
        "with 'pareto_front', 'target_objective', and 'generators'."
    )
