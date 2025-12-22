import numpy as np
import plotly.graph_objects as go

from .layout_config import PARETO_MARKER, TARGET_MARKER


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
