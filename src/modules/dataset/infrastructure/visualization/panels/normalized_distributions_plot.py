from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..helpers.pdf_1d import add_pdf1d

_OBJ_TRAIN = "#2980b9"
_OBJ_TEST = "#5dade2"
_DEC_TRAIN = "#d35400"
_DEC_TEST = "#e59866"


def create_normalized_decision_pdf_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 1D PDF plots for normalized decisions."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Norm x1 PDF", "Norm x2 PDF"))

    # x1 (col 0)
    _add_pdf_overlay(
        fig, data, ["X_train", "X_test"], 0, 1, 1, [_DEC_TRAIN, _DEC_TEST], "Norm $x_1$"
    )

    # x2 (col 1)
    _add_pdf_overlay(
        fig, data, ["X_train", "X_test"], 1, 1, 2, [_DEC_TRAIN, _DEC_TEST], "Norm $x_2$"
    )

    fig.update_layout(
        title="<b>Normalized Decision PDFs</b>",
        template="plotly_white",
        height=500,
        width=1000,
        showlegend=True,
    )
    return fig


def create_normalized_objective_pdf_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 1D PDF plots for normalized objectives."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Norm y1 PDF", "Norm y2 PDF"))

    # y1 (col 0)
    _add_pdf_overlay(
        fig, data, ["y_train", "y_test"], 0, 1, 1, [_OBJ_TRAIN, _OBJ_TEST], "Norm $y_1$"
    )

    # y2 (col 1)
    _add_pdf_overlay(
        fig, data, ["y_train", "y_test"], 1, 1, 2, [_OBJ_TRAIN, _OBJ_TEST], "Norm $y_2$"
    )

    fig.update_layout(
        title="<b>Normalized Objective PDFs</b>",
        template="plotly_white",
        height=500,
        width=1000,
        showlegend=True,
    )
    return fig


def _add_pdf_overlay(fig, data, keys, data_col, row, plot_col, colors, label):
    for k, color in zip(keys, colors):
        arr = _get_2d(data, k)
        if arr.size and arr.shape[1] > data_col:
            v = arr[:, data_col]
            add_pdf1d(fig, row, plot_col, v, label, color=color)


def _get_2d(data: dict[str, Any], key: str):
    arr = np.asarray(data.get(key, []))
    if arr.size == 0:
        return np.empty((0, 2))
    return arr[:, :2] if arr.ndim == 2 else arr.reshape(-1, 2)
