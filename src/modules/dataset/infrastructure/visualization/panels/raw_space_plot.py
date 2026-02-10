import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_2d import add_scatter_overlay


def create_raw_decision_space_figure(
    pareto_set: np.ndarray, historical_solutions: np.ndarray
) -> go.Figure:
    """Creates scatter plot for Raw Decision Space (x1 vs x2)."""
    fig = make_subplots(rows=1, cols=1)

    # Base Pareto Set
    add_scatter_overlay(
        fig,
        1,
        1,
        pareto_set[:, 0],
        pareto_set[:, 1],
        name="Pareto Set",
        symbol="circle",
        size=7,
        opacity=1.0,
        color="#d35400",  # Pumpkin
    )

    # Historical Decisions Overlay
    add_scatter_overlay(
        fig,
        1,
        1,
        historical_solutions[:, 0],
        historical_solutions[:, 1],
        name="Historical (Decisions)",
        color="#bdc3c7",  # Silver
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
    fig = make_subplots(rows=1, cols=1)

    # Base Pareto Front
    add_scatter_overlay(
        fig,
        1,
        1,
        pareto_front[:, 0],
        pareto_front[:, 1],
        name="Pareto Front",
        symbol="circle",
        size=7,
        opacity=1.0,
        color="#2980b9",  # Belize Hole
    )

    # Historical Objectives Overlay
    add_scatter_overlay(
        fig,
        1,
        1,
        historical_objectives[:, 0],
        historical_objectives[:, 1],
        name="Historical (Objectives)",
        color="#bdc3c7",  # Silver
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
