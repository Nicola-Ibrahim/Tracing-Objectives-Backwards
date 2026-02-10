import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.pdf_2d import add_pdf2d

_DEC_COLORSCALE = "Oranges"
_OBJ_COLORSCALE = "Blues"


def create_normalized_decision_density_figure(X_train: np.ndarray) -> go.Figure:
    """Creates 2D density plot for normalized decisions."""
    fig = make_subplots(rows=1, cols=1)

    # Concatenate train and test for density
    if X_train.size:
        add_pdf2d(
            fig,
            1,
            1,
            X_train[:, 0],
            X_train[:, 1],
            "Norm $x_1$",
            "Norm $x_2$",
            colorscale=_DEC_COLORSCALE,
            show_points=False,
        )

    fig.update_layout(
        title="<b>Normalized Decision Density</b>",
        template="plotly_white",
        height=600,
        width=700,
    )
    return fig


def create_normalized_objective_density_figure(y_train: np.ndarray) -> go.Figure:
    """Creates 2D density plot for normalized objectives."""
    fig = make_subplots(rows=1, cols=1)

    # Concatenate train and test for density
    if y_train.size:
        add_pdf2d(
            fig,
            1,
            1,
            y_train[:, 0],
            y_train[:, 1],
            "Norm $y_1$",
            "Norm $y_2$",
            colorscale=_OBJ_COLORSCALE,
            show_points=False,
        )

    fig.update_layout(
        title="<b>Normalized Objective Density</b>",
        template="plotly_white",
        height=600,
        width=700,
    )
    return fig
