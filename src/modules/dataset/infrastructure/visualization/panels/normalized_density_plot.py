from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..helpers.pdf_2d import add_pdf2d

_DEC_COLORSCALE = "Oranges"
_OBJ_COLORSCALE = "Blues"


def create_normalized_decision_density_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 2D density plot for normalized decisions."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1)

    # Concatenate train and test for density
    d = _concat_mats(data, ["X_train", "X_test"])
    if d.size:
        add_pdf2d(
            fig,
            1,
            1,
            d[:, 0],
            d[:, 1],
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


def create_normalized_objective_density_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 2D density plot for normalized objectives."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1)

    # Concatenate train and test for density
    d = _concat_mats(data, ["y_train", "y_test"])
    if d.size:
        add_pdf2d(
            fig,
            1,
            1,
            d[:, 0],
            d[:, 1],
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


def _concat_mats(data, keys):
    mats = [_get_2d(data, k) for k in keys]
    mats = [m for m in mats if m.size]
    return np.vstack(mats) if mats else np.empty((0, 2))


def _get_2d(data: dict[str, Any], key: str):
    arr = np.asarray(data.get(key, []))
    if arr.size == 0:
        return np.empty((0, 2))
    return arr[:, :2] if arr.ndim == 2 else arr.reshape(-1, 2)
