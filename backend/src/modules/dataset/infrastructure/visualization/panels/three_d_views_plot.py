import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_3d import add_3d_overlay


def create_3d_decision_context_figure(
    X_raw: np.ndarray, y_raw: np.ndarray
) -> go.Figure:
    """
    Creates 3D plots showing Decision Space (x1, x2) vs Objective 
    components (y1, y2).
    """

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
        x=X_raw[:, 0],
        y=X_raw[:, 1],
        z=y_raw[:, 0],
        name="Raw Data",
        size=3,
        opacity=0.8,
        color="#d35400",
    )

    # Plot 2: x1, x2, y2
    add_3d_overlay(
        fig=fig,
        row=1,
        col=2,
        x=X_raw[:, 0],
        y=X_raw[:, 1],
        z=y_raw[:, 1],
        name="Raw Data",
        size=3,
        opacity=0.8,
        color="#2980b9",
    )

    fig.update_scenes(
        xaxis=dict(title="x1 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        yaxis=dict(title="x2 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        zaxis=dict(title="y1 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        row=1,
        col=1,
    )
    fig.update_scenes(
        xaxis=dict(title="x1 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        yaxis=dict(title="x2 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        zaxis=dict(title="y2 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text="<b>3D Decision Context (Decisions x Objectives)</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        template="plotly_white",
        height=700,
        width=1200,
        margin=dict(t=100, b=50, l=50, r=50),
    )
    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=18)

    return fig


def create_3d_objective_context_figure(
    X_raw: np.ndarray, y_raw: np.ndarray
) -> go.Figure:
    """
    Creates 3D plots showing Objective Space (y1, y2) vs Decision 
    components (x1, x2).
    """

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
        x=y_raw[:, 0],
        y=y_raw[:, 1],
        z=X_raw[:, 0],
        name="Raw Data",
        size=3,
        opacity=0.8,
        color="#2980b9",
    )

    # Plot 2: y1, y2, x2
    add_3d_overlay(
        fig=fig,
        row=1,
        col=2,
        x=y_raw[:, 0],
        y=y_raw[:, 1],
        z=X_raw[:, 1],
        name="Raw Data",
        size=3,
        opacity=0.8,
        color="#d35400",
    )

    fig.update_scenes(
        xaxis=dict(title="y1 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        yaxis=dict(title="y2 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        zaxis=dict(title="x1 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        row=1,
        col=1,
    )
    fig.update_scenes(
        xaxis=dict(title="y1 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        yaxis=dict(title="y2 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        zaxis=dict(title="x2 (Raw)", title_font=dict(size=16), tickfont=dict(size=12)),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text="<b>3D Objective Context (Objectives x Decisions)</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        template="plotly_white",
        height=700,
        width=1200,
        margin=dict(t=100, b=50, l=50, r=50),
    )
    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=18)

    return fig
