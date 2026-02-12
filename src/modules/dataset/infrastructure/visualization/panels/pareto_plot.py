import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_2d import add_scatter_overlay


def create_pareto_set_figure(pareto_set: np.ndarray) -> go.Figure:
    """Creates scatter plot of pareto set only."""
    fig = make_subplots(rows=1, cols=1)

    add_scatter_overlay(
        fig,
        1,
        1,
        pareto_set[:, 0],
        pareto_set[:, 1],
        name="Pareto Set",
        symbol="circle",
        size=10,
        opacity=1.0,
        color="#d35400",
        show_legend=False,
    )

    fig.update_layout(
        title=dict(
            text="<b>Pareto Set (Decision Space)</b>",
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
    )
    return fig


def create_pareto_front_figure(pareto_front: np.ndarray) -> go.Figure:
    """Creates scatter plot of pareto front only."""
    fig = make_subplots(rows=1, cols=1)

    add_scatter_overlay(
        fig,
        1,
        1,
        pareto_front[:, 0],
        pareto_front[:, 1],
        name="Pareto Front",
        symbol="circle",
        size=10,
        opacity=1.0,
        color="#2980b9",
        show_legend=False,
    )

    fig.update_layout(
        title=dict(
            text="<b>Pareto Front (Objective Space)</b>",
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
    )
    return fig
