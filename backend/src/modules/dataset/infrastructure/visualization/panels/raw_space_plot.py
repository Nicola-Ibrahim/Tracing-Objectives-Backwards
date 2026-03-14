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
        size=8,
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
        size=6,
        opacity=0.5,
    )

    fig.update_layout(
        title=dict(
            text="<b>Decision Space (Raw)</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        xaxis=dict(
            title="$x_1$",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
        ),
        yaxis=dict(
            title="$x_2$",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
        ),
        template="plotly_white",
        height=700,
        width=800,
        margin=dict(t=100, b=100, l=100, r=100),
        legend=dict(font=dict(size=14)),
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
        size=8,
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
        size=6,
        opacity=0.5,
    )

    fig.update_layout(
        title=dict(
            text="<b>Objective Space (Raw)</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        xaxis=dict(
            title="$y_1$",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
        ),
        yaxis=dict(
            title="$y_2$",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
        ),
        template="plotly_white",
        height=700,
        width=800,
        margin=dict(t=100, b=100, l=100, r=100),
        legend=dict(font=dict(size=14)),
    )
    return fig
