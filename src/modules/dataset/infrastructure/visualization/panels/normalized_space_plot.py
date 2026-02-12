import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_2d import add_scatter_overlay


def create_normalized_decision_space_figure(X_train: np.ndarray) -> go.Figure:
    """Creates scatter plot for Normalized Decision Space (x1 vs x2)."""
    fig = make_subplots(rows=1, cols=1)

    # Train (Decisions)
    add_scatter_overlay(
        fig,
        1,
        1,
        X_train[:, 0],
        X_train[:, 1],
        name="Train (Decisions)",
        symbol="circle-open",
        size=6,
        opacity=0.7,
        color="#d35400",
    )

    fig.update_layout(
        title=dict(
            text="<b>Normalized Decisions</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        xaxis=dict(
            title="Norm $x_1$",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
        ),
        yaxis=dict(
            title="Norm $x_2$",
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
    )
    return fig


def create_normalized_objective_space_figure(y_train: np.ndarray) -> go.Figure:
    """Creates scatter plot for Normalized Objective Space (y1 vs y2)."""
    fig = make_subplots(rows=1, cols=1)

    # Train (Objectives)
    add_scatter_overlay(
        fig,
        1,
        1,
        y_train[:, 0],
        y_train[:, 1],
        name="Train (Objectives)",
        symbol="circle-open",
        size=6,
        opacity=0.7,
        color="#2980b9",
    )

    fig.update_layout(
        title=dict(
            text="<b>Normalized Objectives</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        xaxis=dict(
            title="Norm $y_1$",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
        ),
        yaxis=dict(
            title="Norm $y_2$",
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
    )
    return fig
