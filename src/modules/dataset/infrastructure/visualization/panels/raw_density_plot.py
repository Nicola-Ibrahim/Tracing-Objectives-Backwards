from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..helpers.pdf_2d import add_pdf2d

_DEC_COLORSCALE = "Oranges"
_OBJ_COLORSCALE = "Blues"


def create_raw_decision_density_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 2D PDF density plot for raw decisions."""
    fig = go.Figure()

    arr = _get_2d(
        data, "historical_solutions"
    )  # Use historical for density as per original
    if arr.size:
        # Note: add_pdf2d expects a figure and adds a Trace. defined in pdf_2d.py
        # It typically adds a contour/heatmap.
        # Since we are not using subplots, row/col can be None if the helper supports it,
        # or we assume 1,1. Let's check pdf_2d usage or wrap it.
        # Original usage: add_pdf2d(fig, row, col, x, y, ...)
        # I'll assume 1,1 for now or pass None if compatible.
        # Actually standard plotly add_trace works on fig.

        # We might need to use make_subplots(rows=1, cols=1) to be safe with helper helpers
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=1)

        add_pdf2d(
            fig,
            1,
            1,
            arr[:, 0],
            arr[:, 1],
            "$x_1$ (Raw)",
            "$x_2$ (Raw)",
            colorscale=_DEC_COLORSCALE,
            show_points=False,
        )

    fig.update_layout(
        title="<b>Raw Decision Density</b>",
        template="plotly_white",
        height=600,
        width=700,
    )
    return fig


def create_raw_objective_density_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 2D PDF density plot for raw objectives."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1)

    arr = _get_2d(data, "historical_objectives")
    if arr.size:
        add_pdf2d(
            fig,
            1,
            1,
            arr[:, 0],
            arr[:, 1],
            "$y_1$ (Raw)",
            "$y_2$ (Raw)",
            colorscale=_OBJ_COLORSCALE,
            show_points=False,
        )

    fig.update_layout(
        title="<b>Raw Objective Density</b>",
        template="plotly_white",
        height=600,
        width=700,
    )
    return fig


def _get_2d(data: dict[str, Any], key: str):
    arr = np.asarray(data.get(key, []))
    if arr.size == 0:
        return np.empty((0, 2))
    return arr[:, :2] if arr.ndim == 2 else arr.reshape(-1, 2)
