from typing import Any

import plotly.graph_objects as go

from ..helpers.scatter_2d import add_scatter_overlay

_DEC_TRAIN = "#d35400"  # Pumpkin
_OBJ_TRAIN = "#2980b9"  # Belize Hole
_HISTORY_COLOR = "#bdc3c7"  # Silver
_DEFAULT_COLOR = "#888888"  # Grey fallback


def create_raw_decision_space_figure(data: dict[str, Any]) -> go.Figure:
    """Creates scatter plot for Raw Decision Space (x1 vs x2)."""
    fig = go.Figure()

    # Base Pareto Set
    _add_scatter_layer(
        fig,
        data,
        "pareto_set",
        name="Pareto Set",
        color=_DEC_TRAIN,
        symbol="circle",
        size=7,
    )

    # Historical Decisions Overlay
    _add_scatter_layer(
        fig,
        data,
        "historical_solutions",
        name="Historical (Decisions)",
        color=_HISTORY_COLOR,
        symbol="cross",
        size=5,
        opacity=0.5,
    )

    fig.update_layout(
        title="<b>Decision Space (Raw)</b>",
        xaxis_title="$x_1$",
        yaxis_title="$x_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig


def create_raw_objective_space_figure(data: dict[str, Any]) -> go.Figure:
    """Creates scatter plot for Raw Objective Space (y1 vs y2)."""
    fig = go.Figure()

    # Base Pareto Front
    _add_scatter_layer(
        fig,
        data,
        "pareto_front",
        name="Pareto Front",
        color=_OBJ_TRAIN,
        symbol="circle",
        size=7,
    )

    # Historical Objectives Overlay
    _add_scatter_layer(
        fig,
        data,
        "historical_objectives",
        name="Historical (Objectives)",
        color=_HISTORY_COLOR,
        symbol="cross",
        size=5,
        opacity=0.5,
    )

    fig.update_layout(
        title="<b>Objective Space (Raw)</b>",
        xaxis_title="$y_1$",
        yaxis_title="$y_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig


def _add_scatter_layer(fig, data, key, name, color, symbol, size, opacity=1.0):
    arr = _get_2d(data, key)
    if arr.size:
        add_scatter_overlay(
            fig,
            None,
            None,  # row/col not needed for single plot
            arr[:, 0],
            arr[:, 1],
            name=name,
            symbol=symbol,
            size=size,
            opacity=opacity,
            color=color,
        )


def _get_2d(data: dict[str, Any], key: str):
    import numpy as np

    arr = np.asarray(data.get(key, []))
    if arr.size == 0:
        return np.empty((0, 2))
    if arr.ndim == 1:
        n2 = arr.size // 2
        arr = arr.reshape(n2, 2)
    return arr[:, :2]
