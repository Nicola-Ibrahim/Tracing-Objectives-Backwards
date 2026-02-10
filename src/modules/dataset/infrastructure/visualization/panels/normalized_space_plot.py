import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_2d import add_scatter_overlay

_OBJ_COLOR = "#2980b9"
_DEC_COLOR = "#d35400"


def create_normalized_decision_space_figure(X_train: np.ndarray) -> go.Figure:
    """Creates scatter plot for Normalized Decision Space (x1 vs x2)."""
    fig = make_subplots(rows=1, cols=1)

    # Train (Decisions)
    if X_train.size:
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
            color=_DEC_COLOR,
        )

    fig.update_layout(
        title="<b>Normalized Decisions</b>",
        xaxis_title="Norm $x_1$",
        yaxis_title="Norm $x_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig


def create_normalized_objective_space_figure(y_train: np.ndarray) -> go.Figure:
    """Creates scatter plot for Normalized Objective Space (y1 vs y2)."""
    fig = make_subplots(rows=1, cols=1)

    # Train (Objectives)
    if y_train.size:
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
            color=_OBJ_COLOR,
        )

    fig.update_layout(
        title="<b>Normalized Objectives</b>",
        xaxis_title="Norm $y_1$",
        yaxis_title="Norm $y_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig
