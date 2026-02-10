import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_3d import add_3d_overlay


def create_3d_decision_context_figure(
    X_train: np.ndarray, y_train: np.ndarray
) -> go.Figure:
    """Creates 3D plots showing Decision Space (x1, x2) vs Objective components (y1, y2)."""

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("(x1, x2, y1)", "(x1, x2, y2)"),
    )

    # Plot 1: x1, x2, y1
    add_3d_overlay(
        fig=fig,
        row=1,
        col=1,
        x=X_train[:, 0],
        y=X_train[:, 1],
        z=y_train[:, 0],
        name="Train",
        size=3,
        opacity=0.8,
        color="#d35400",
    )

    # Plot 2: x1, x2, y2
    add_3d_overlay(
        fig=fig,
        row=1,
        col=2,
        x=X_train[:, 0],
        y=X_train[:, 1],
        z=y_train[:, 1],
        name="Train",
        size=3,
        opacity=0.8,
        color="#2980b9",
    )

    fig.update_scenes(
        xaxis_title="x1 (norm)",
        yaxis_title="x2 (norm)",
        zaxis_title="y1 (norm)",
        row=1,
        col=1,
    )
    fig.update_scenes(
        xaxis_title="x1 (norm)",
        yaxis_title="x2 (norm)",
        zaxis_title="y2 (norm)",
        row=1,
        col=2,
    )

    fig.update_layout(
        title="<b>3D Decision Context (Decisions x Objectives)</b>",
        template="plotly_white",
        height=600,
        width=1200,
    )
    return fig


def create_3d_objective_context_figure(
    X_train: np.ndarray, y_train: np.ndarray
) -> go.Figure:
    """Creates 3D plots showing Objective Space (y1, y2) vs Decision components (x1, x2)."""

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("(y1, y2, x1)", "(y1, y2, x2)"),
    )

    # Plot 1: y1, y2, x1
    add_3d_overlay(
        fig=fig,
        row=1,
        col=1,
        x=y_train[:, 0],
        y=y_train[:, 1],
        z=X_train[:, 0],
        name="Train",
        size=3,
        opacity=0.8,
        color="#2980b9",
    )

    # Plot 2: y1, y2, x2
    add_3d_overlay(
        fig=fig,
        row=1,
        col=2,
        x=y_train[:, 0],
        y=y_train[:, 1],
        z=X_train[:, 1],
        name="Train",
        size=3,
        opacity=0.8,
        color="#d35400",
    )

    fig.update_scenes(
        xaxis_title="y1 (norm)",
        yaxis_title="y2 (norm)",
        zaxis_title="x1 (norm)",
        row=1,
        col=1,
    )
    fig.update_scenes(
        xaxis_title="y1 (norm)",
        yaxis_title="y2 (norm)",
        zaxis_title="x2 (norm)",
        row=1,
        col=2,
    )

    fig.update_layout(
        title="<b>3D Objective Context (Objectives x Decisions)</b>",
        template="plotly_white",
        height=600,
        width=1200,
    )
    return fig
