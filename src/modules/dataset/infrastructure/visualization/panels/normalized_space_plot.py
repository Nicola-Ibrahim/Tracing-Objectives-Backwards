from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..helpers.scatter_2d import add_scatter_overlay

_OBJ_TRAIN = "#2980b9"
_OBJ_TEST = "#5dade2"
_DEC_TRAIN = "#d35400"
_DEC_TEST = "#e59866"
_DEFAULT_COLOR = "#888888"


def create_normalized_decision_space_figure(data: dict[str, Any]) -> go.Figure:
    """Creates scatter plot for Normalized Decision Space (x1 vs x2)."""
    fig = go.Figure()

    # Train (Decisions)
    _add_series(
        fig, data, "X_train", "Train (Decisions)", _DEC_TRAIN, "circle-open", 6, 0.7
    )

    # Test (Decisions)
    _add_series(fig, data, "X_test", "Test (Decisions)", _DEC_TEST, "x", 7, 0.9)

    fig.update_layout(
        title="<b>Normalized Decisions</b>",
        xaxis_title="Norm $x_1$",
        yaxis_title="Norm $x_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig


def create_normalized_objective_space_figure(data: dict[str, Any]) -> go.Figure:
    """Creates scatter plot for Normalized Objective Space (y1 vs y2)."""
    fig = go.Figure()

    # Train (Objectives)
    _add_series(
        fig, data, "y_train", "Train (Objectives)", _OBJ_TRAIN, "circle-open", 6, 0.7
    )

    # Test (Objectives)
    _add_series(fig, data, "y_test", "Test (Objectives)", _OBJ_TEST, "x", 7, 0.9)

    fig.update_layout(
        title="<b>Normalized Objectives</b>",
        xaxis_title="Norm $y_1$",
        yaxis_title="Norm $y_2$",
        template="plotly_white",
        height=600,
        width=800,
    )
    return fig


def _add_series(fig, data, key, name, color, symbol, size, opacity):
    arr = _get_2d(data, key)
    if arr.size:
        add_scatter_overlay(
            fig,
            None,
            None,
            arr[:, 0],
            arr[:, 1],
            name=name,
            symbol=symbol,
            size=size,
            opacity=opacity,
            color=color,
        )


def _get_2d(data: dict[str, Any], key: str):
    arr = np.asarray(data.get(key, []))
    if arr.size == 0:
        return np.empty((0, 2))
    return arr[:, :2] if arr.ndim == 2 else arr.reshape(-1, 2)
