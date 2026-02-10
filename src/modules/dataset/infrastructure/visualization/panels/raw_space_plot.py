import numpy as np
import plotly.graph_objects as go

from ..helpers.scatter_2d import add_scatter_overlay

_DEC_COLOR = "#d35400"  # Pumpkin
_OBJ_COLOR = "#2980b9"  # Belize Hole
_HISTORY_COLOR = "#bdc3c7"  # Silver


def create_raw_decision_space_figure(
    pareto_set: np.ndarray, historical_solutions: np.ndarray
) -> go.Figure:
    """Creates scatter plot for Raw Decision Space (x1 vs x2)."""
    fig = go.Figure()

    # Base Pareto Set
    if pareto_set.size:
        add_scatter_overlay(
            fig,
            None,
            None,  # row/col not needed for single plot
            pareto_set[:, 0],
            pareto_set[:, 1],
            name="Pareto Set",
            symbol="circle",
            size=7,
            opacity=1.0,
            color=_DEC_COLOR,
        )

    # Historical Decisions Overlay
    if historical_solutions.size:
        add_scatter_overlay(
            fig,
            None,
            None,
            historical_solutions[:, 0],
            historical_solutions[:, 1],
            name="Historical (Decisions)",
            color=_HISTORY_COLOR,
            symbol="cross",
            size=5,
            opacity=0.5,
        )

    fig.update_layout(
        title="<b>Decision Space (Raw)</b>",
        xaxis_title="$x_1$",
        yaxis_title="$x_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig


def create_raw_objective_space_figure(
    pareto_front: np.ndarray, historical_objectives: np.ndarray
) -> go.Figure:
    """Creates scatter plot for Raw Objective Space (y1 vs y2)."""
    fig = go.Figure()

    # Base Pareto Front
    if pareto_front.size:
        add_scatter_overlay(
            fig,
            None,
            None,
            pareto_front[:, 0],
            pareto_front[:, 1],
            name="Pareto Front",
            symbol="circle",
            size=7,
            opacity=1.0,
            color=_OBJ_COLOR,
        )

    # Historical Objectives Overlay
    if historical_objectives.size:
        add_scatter_overlay(
            fig,
            None,
            None,
            historical_objectives[:, 0],
            historical_objectives[:, 1],
            name="Historical (Objectives)",
            color=_HISTORY_COLOR,
            symbol="cross",
            size=5,
            opacity=0.5,
        )

    fig.update_layout(
        title="<b>Objective Space (Raw)</b>",
        xaxis_title="$y_1$",
        yaxis_title="$y_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig
