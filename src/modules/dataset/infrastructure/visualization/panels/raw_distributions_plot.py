from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..helpers.pdf_1d import add_pdf1d

_DEC_TRAIN = "#d35400"
_OBJ_TRAIN = "#2980b9"


def create_raw_decision_distributions_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 1D KDE plots for raw decision variables (x1, x2)."""
    fig = go.Figure()  # Using make_subplots would be better if we want side-by-side,
    # but user asked for "save each individually".
    # Actually, usually distribution plots are grouped.
    # Let's keep them as a pair in one figure for "Distributions" or split them?
    # User said "split the plots... save each one individually".
    # I will split x1 and x2 into separate plots or subplots in one figure?
    # "save each plot individually with specific name" -> likely implies one file per view.
    # But having x1 and x2 separate files might be too granular?
    # "Decision and Objective space shouldn't be in one function".
    # I'll group x1 and x2 into "Decision Distributions".

    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("x1 KDE", "x2 KDE"))

    arr = _get_2d(data, "pareto_set")
    if arr.size:
        add_pdf1d(fig, 1, 1, arr[:, 0], "$x_1$", color=_DEC_TRAIN)
        add_pdf1d(fig, 1, 2, arr[:, 1], "$x_2$", color=_DEC_TRAIN)

    fig.update_layout(
        title="<b>Raw Decision Distributions</b>",
        template="plotly_white",
        height=500,
        width=1000,
        showlegend=False,
    )
    return fig


def create_raw_objective_distributions_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 1D KDE plots for raw objective variables (y1, y2)."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("y1 KDE", "y2 KDE"))

    arr = _get_2d(data, "pareto_front")
    if arr.size:
        add_pdf1d(fig, 1, 1, arr[:, 0], "$y_1$", color=_OBJ_TRAIN)
        add_pdf1d(fig, 1, 2, arr[:, 1], "$y_2$", color=_OBJ_TRAIN)

    fig.update_layout(
        title="<b>Raw Objective Distributions</b>",
        template="plotly_white",
        height=500,
        width=1000,
        showlegend=False,
    )
    return fig


def _get_2d(data: dict[str, Any], key: str):
    arr = np.asarray(data.get(key, []))
    if arr.size == 0:
        return np.empty((0, 2))
    if arr.ndim == 1:
        n2 = arr.size // 2
        arr = arr.reshape(n2, 2)
    return arr[:, :2]
