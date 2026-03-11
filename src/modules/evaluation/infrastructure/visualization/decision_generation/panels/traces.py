import numpy as np
import plotly.graph_objects as go

from .layout_config import BEST_MARKER, PARETO_MARKER, TARGET_MARKER


def add_prediction_trace(
    fig: go.Figure, run: dict, idx: int, row: int, col: int, color: str
) -> None:
    """Add the scatter trace for model predictions."""
    name = str(run.get("name", f"model_{idx}"))
    predicted = np.asarray(run.get("predicted_objectives"), dtype=float)

    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    if predicted.shape[1] < 2:
        raise ValueError(
            f"Predicted objectives for '{name}' must have at least two columns."
        )

    fig.add_trace(
        go.Scatter(
            x=predicted[:, 0],
            y=predicted[:, 1],
            mode="markers",
            name=name,
            marker=dict(color=color, size=6, opacity=0.7),
            showlegend=True,
        ),
        row=row,
        col=col,
    )


def add_pareto_front_trace(
    fig: go.Figure, pareto_front: np.ndarray, show_legend: bool, row: int, col: int
) -> None:
    """Add the Pareto front background trace."""
    fig.add_trace(
        go.Scatter(
            x=pareto_front[:, 0],
            y=pareto_front[:, 1],
            mode="markers",
            name="Pareto Front",
            marker=PARETO_MARKER,
            showlegend=show_legend,
            legendgroup="pareto",
        ),
        row=row,
        col=col,
    )


def add_target_trace(
    fig: go.Figure, target: np.ndarray, show_legend: bool, row: int, col: int
) -> None:
    """Add the target objective star trace."""
    fig.add_trace(
        go.Scatter(
            x=[target[0]],
            y=[target[1]],
            mode="markers",
            name="Target Objective",
            marker=TARGET_MARKER,
            showlegend=show_legend,
            legendgroup="target",
        ),
        row=row,
        col=col,
    )


def add_decision_trace(
    fig: go.Figure, run: dict, idx: int, row: int, col: int, color: str
) -> None:
    """Add decision space scatter trace."""
    name = str(run.get("name", f"model_{idx}"))
    decisions = np.asarray(run.get("decisions"), dtype=float)

    # Use first two dimensions for 2D plot
    fig.add_trace(
        go.Scatter(
            x=decisions[:, 0],
            y=decisions[:, 1],
            mode="markers",
            name=f"{name} Decisions",
            marker=dict(color=color, size=6, opacity=0.7),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def add_pareto_set_trace(
    fig: go.Figure, pareto_set: np.ndarray, show_legend: bool, row: int, col: int
) -> None:
    """Add the Pareto set background trace to decision space."""
    fig.add_trace(
        go.Scatter(
            x=pareto_set[:, 0],
            y=pareto_set[:, 1],
            mode="markers",
            name="Pareto Set",
            marker=PARETO_MARKER,
            showlegend=show_legend,
            legendgroup="pareto_set",
        ),
        row=row,
        col=col,
    )


def add_best_decision_trace(
    fig: go.Figure, run: dict, row: int, col: int, color: str
) -> None:
    """Highlight the best decision point in the decision space."""
    best_decision = run.get("best_decision")
    if best_decision is None:
        return

    fig.add_trace(
        go.Scatter(
            x=[best_decision[0]],
            y=[best_decision[1]],
            mode="markers",
            name="Best Decision",
            marker=dict(color=color, **BEST_MARKER),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def add_objective_connection_trace(
    fig: go.Figure, run: dict, target: np.ndarray, row: int, col: int, color: str
) -> None:
    """Draw a connection line between the target objective and the best generator objective."""
    best_obj = run.get("best_objective")
    if best_obj is None:
        return

    fig.add_trace(
        go.Scatter(
            x=[target[0], best_obj[0]],
            y=[target[1], best_obj[1]],
            mode="lines",
            name="Target Link",
            line=dict(color=color, width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def add_best_objective_trace(
    fig: go.Figure, run: dict, row: int, col: int, color: str
) -> None:
    """Highlight the best generated objective in the objective space."""
    best_obj = run.get("best_objective")
    if best_obj is None:
        return

    fig.add_trace(
        go.Scatter(
            x=[best_obj[0]],
            y=[best_obj[1]],
            mode="markers",
            name="Best Objective",
            marker=dict(
                color="red",
                symbol="x",
                size=10,
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
