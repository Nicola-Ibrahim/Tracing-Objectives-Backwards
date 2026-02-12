import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.pdf_1d import add_pdf1d

_DEC_COLOR = "#d35400"
_OBJ_COLOR = "#2980b9"


def create_raw_decision_distributions_figure(pareto_set: np.ndarray) -> go.Figure:
    """Creates 1D KDE plots for raw decision variables (x1, x2)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("x1 KDE", "x2 KDE"))

    # x1
    add_pdf1d(fig, 1, 1, pareto_set[:, 0], "$x_1$", color=_DEC_COLOR)
    # x2
    if pareto_set.shape[1] > 1:
        add_pdf1d(fig, 1, 2, pareto_set[:, 1], "$x_2$", color=_DEC_COLOR)

    fig.update_layout(
        title=dict(
            text="<b>Raw Decision Distributions</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        template="plotly_white",
        height=600,
        width=1200,
        margin=dict(t=120, b=100, l=100, r=100),
        showlegend=False,
    )
    fig.update_xaxes(
        title_font=dict(size=18),
        tickfont=dict(size=14),
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=False,
    )
    fig.update_yaxes(
        title_font=dict(size=18),
        tickfont=dict(size=14),
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=False,
    )
    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=20)
    return fig


def create_raw_objective_distributions_figure(pareto_front: np.ndarray) -> go.Figure:
    """Creates 1D KDE plots for raw objective variables (y1, y2)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("y1 KDE", "y2 KDE"))

    if pareto_front.size:
        # y1
        add_pdf1d(fig, 1, 1, pareto_front[:, 0], "$y_1$", color=_OBJ_COLOR)
        # y2
        if pareto_front.shape[1] > 1:
            add_pdf1d(fig, 1, 2, pareto_front[:, 1], "$y_2$", color=_OBJ_COLOR)

    fig.update_layout(
        title=dict(
            text="<b>Raw Objective Distributions</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        template="plotly_white",
        height=600,
        width=1200,
        margin=dict(t=120, b=100, l=100, r=100),
        showlegend=False,
    )
    fig.update_xaxes(
        title_font=dict(size=18),
        tickfont=dict(size=14),
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=False,
    )
    fig.update_yaxes(
        title_font=dict(size=18),
        tickfont=dict(size=14),
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=False,
    )
    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=20)
    return fig
