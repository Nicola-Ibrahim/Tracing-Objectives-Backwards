import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.pdf_2d import add_pdf2d


def create_normalized_decision_density_figure(X_train: np.ndarray) -> go.Figure:
    """Creates 2D density plot for normalized decisions."""
    fig = make_subplots(rows=1, cols=1)

    # Concatenate train and test for density
    add_pdf2d(
        fig,
        1,
        1,
        X_train[:, 0],
        X_train[:, 1],
        "Norm $x_1$",
        "Norm $x_2$",
        colorscale="Oranges",
        show_points=False,
    )

    fig.update_layout(
        title=dict(
            text="<b>Normalized Decision Density</b>",
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


def create_normalized_objective_density_figure(y_train: np.ndarray) -> go.Figure:
    """Creates 2D density plot for normalized objectives."""
    fig = make_subplots(rows=1, cols=1)

    # Concatenate train and test for density
    add_pdf2d(
        fig,
        1,
        1,
        y_train[:, 0],
        y_train[:, 1],
        "Norm $y_1$",
        "Norm $y_2$",
        colorscale="Blues",
        show_points=False,
    )

    fig.update_layout(
        title=dict(
            text="<b>Normalized Objective Density</b>",
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
