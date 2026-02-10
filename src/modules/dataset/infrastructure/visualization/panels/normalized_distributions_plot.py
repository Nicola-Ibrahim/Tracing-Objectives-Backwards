import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.pdf_1d import add_pdf1d


def create_normalized_decision_pdf_figure(X_train: np.ndarray) -> go.Figure:
    """Creates 1D PDF plots for normalized decisions."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Norm x1 PDF", "Norm x2 PDF"))

    # x1 (col 0)
    if X_train.size and X_train.shape[1] > 0:
        add_pdf1d(fig, 1, 1, X_train[:, 0], "Norm $x_1$", color="#d35400")

    # x2 (col 1)
    if X_train.size and X_train.shape[1] > 1:
        add_pdf1d(fig, 1, 2, X_train[:, 1], "Norm $x_2$", color="#d35400")

    fig.update_layout(
        title="<b>Normalized Decision PDFs</b>",
        template="plotly_white",
        height=500,
        width=1000,
        showlegend=True,
    )
    return fig


def create_normalized_objective_pdf_figure(y_train: np.ndarray) -> go.Figure:
    """Creates 1D PDF plots for normalized objectives."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Norm y1 PDF", "Norm y2 PDF"))

    # y1 (col 0)
    if y_train.size and y_train.shape[1] > 0:
        add_pdf1d(fig, 1, 1, y_train[:, 0], "Norm $y_1$", color="#2980b9")

    # y2 (col 1)
    if y_train.size and y_train.shape[1] > 1:
        add_pdf1d(fig, 1, 2, y_train[:, 1], "Norm $y_2$", color="#2980b9")

    fig.update_layout(
        title="<b>Normalized Objective PDFs</b>",
        template="plotly_white",
        height=500,
        width=1000,
        showlegend=True,
    )
    return fig
