from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_2d import add_scatter_overlay

_DEC_COLOR = "#d35400"
_OBJ_COLOR = "#2980b9"


def create_pareto_set_figure(pareto_set: np.ndarray) -> go.Figure:
    """Creates scatter plot of pareto set only."""
    fig = make_subplots(rows=1, cols=1)

    if pareto_set.size:
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
            color=_DEC_COLOR,
        )

    fig.update_layout(
        title="<b>Pareto Set</b>",
        xaxis_title="$x_1$",
        yaxis_title="$x_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig


def create_pareto_front_figure(pareto_front: np.ndarray) -> go.Figure:
    """Creates scatter plot of pareto front only."""
    fig = make_subplots(rows=1, cols=1)

    if pareto_front.size:
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
            color=_OBJ_COLOR,
        )

    fig.update_layout(
        title="<b>Pareto Front</b>",
        xaxis_title="$y_1$",
        yaxis_title="$y_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig
