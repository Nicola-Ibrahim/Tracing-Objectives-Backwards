import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.pdf_2d import add_pdf2d


def create_raw_decision_density_figure(historical_solutions: np.ndarray) -> go.Figure:
    """Creates 2D PDF density plot for raw decisions."""
    fig = make_subplots(rows=1, cols=1)

    add_pdf2d(
        fig,
        1,
        1,
        historical_solutions[:, 0],
        historical_solutions[:, 1],
        "$x_1$ (Raw)",
        "$x_2$ (Raw)",
        colorscale="Oranges",
        show_points=False,
    )

    fig.update_layout(
        title=dict(
            text="<b>Raw Decision Density</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        template="plotly_white",
        height=700,
        width=800,
        margin=dict(t=100, b=100, l=100, r=100),
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
    return fig


def create_raw_objective_density_figure(historical_objectives: np.ndarray) -> go.Figure:
    """Creates 2D PDF density plot for raw objectives."""
    fig = make_subplots(rows=1, cols=1)

    add_pdf2d(
        fig,
        1,
        1,
        historical_objectives[:, 0],
        historical_objectives[:, 1],
        "$y_1$ (Raw)",
        "$y_2$ (Raw)",
        colorscale="Blues",
        show_points=False,
    )

    fig.update_layout(
        title=dict(
            text="<b>Raw Objective Density</b>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        template="plotly_white",
        height=700,
        width=800,
        margin=dict(t=100, b=100, l=100, r=100),
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
    return fig
