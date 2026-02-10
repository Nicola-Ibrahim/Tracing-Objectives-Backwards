from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_3d import add_3d_overlay

_DEC_TRAIN = "#d35400"
_DEC_TEST = "#e59866"
_OBJ_TRAIN = "#2980b9"
_OBJ_TEST = "#5dade2"


def create_3d_decision_context_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 3D plots showing Decision Space (x1, x2) vs Objective components (y1, y2)."""

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("(x1, x2, y1)", "(x1, x2, y2)"),
    )

    # Plot 1: x1, x2, y1
    _add_3d_series(
        fig=fig,
        data=data,
        row=1,
        col=1,
        x_col=0,
        y_col=1,
        z_col=0,
        z_label="y1 (norm)",
    )  # z uses y1 (col 0 of objectives)

    # Plot 2: x1, x2, y2
    _add_3d_series(
        fig=fig,
        data=data,
        row=1,
        col=2,
        x_col=0,
        y_col=1,
        z_col=1,
        z_label="y2 (norm)",
    )  # z uses y2 (col 1 of objectives)

    fig.update_layout(
        title="<b>3D Decision Context (Decisions x Objectives)</b>",
        template="plotly_white",
        height=600,
        width=1200,
    )
    return fig


def create_3d_objective_context_figure(data: dict[str, Any]) -> go.Figure:
    """Creates 3D plots showing Objective Space (y1, y2) vs Decision components (x1, x2)."""

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("(y1, y2, x1)", "(y1, y2, x2)"),
    )

    # Plot 1: y1, y2, x1 (inverse mapping check)
    # x=y1, y=y2, z=x1
    _add_3d_series_inverse(
        fig=fig,
        data=data,
        row=1,
        col=1,
        x_col=0,
        y_col=1,
        z_col=0,
        z_label="x1 (norm)",
    )

    # Plot 2: y1, y2, x2
    # x=y1, y=y2, z=x2
    _add_3d_series_inverse(
        fig=fig,
        data=data,
        row=1,
        col=2,
        x_col=0,
        y_col=1,
        z_col=1,
        z_label="x2 (norm)",
    )

    fig.update_layout(
        title="<b>3D Objective Context (Objectives x Decisions)</b>",
        template="plotly_white",
        height=600,
        width=1200,
    )
    return fig


def _add_3d_series(
    fig: go.Figure,
    data: dict[str, Any],
    row: int,
    col: int,
    x_col: int,
    y_col: int,
    z_col: int,
    z_label: str,
):
    # Train
    _plot_3d(
        fig=fig,
        data=data,
        x_key="X_train",
        y_key="X_train",
        z_key="y_train",
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        row=row,
        col=col,
        name="Train",
        color=_DEC_TRAIN,
    )
    # Test
    _plot_3d(
        fig=fig,
        data=data,
        x_key="X_test",
        y_key="X_test",
        z_key="y_test",
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        row=row,
        col=col,
        name="Test",
        color=_DEC_TEST,
    )

    fig.update_scenes(
        xaxis_title="x1 (norm)",
        yaxis_title="x2 (norm)",
        zaxis_title=z_label,
        row=row,
        col=col,
    )


def _add_3d_series_inverse(
    fig: go.Figure,
    data: dict[str, Any],
    row: int,
    col: int,
    x_col: int,
    y_col: int,
    z_col: int,
    z_label: str,
):
    # Mapping confirmed from VisualizeDatasetCommandHandler:
    # X_train/X_test = Decisions
    # y_train/y_test = Objectives

    # For inverse mapping (Objective Space input):
    # x-axis = y1 (Objective 1) -> y_train[:,0]
    # y-axis = y2 (Objective 2) -> y_train[:,1]
    # z-axis = x1 (Decision 1)  -> X_train[:,0]

    # helper _plot_3d expects key for x, y, z sources.

    # Train
    _plot_3d(
        fig=fig,
        data=data,
        x_key="y_train",
        y_key="y_train",
        z_key="X_train",
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        row=row,
        col=col,
        name="Train",
        color=_OBJ_TRAIN,
    )
    # Test
    _plot_3d(
        fig=fig,
        data=data,
        x_key="y_test",
        y_key="y_test",
        z_key="X_test",
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        row=row,
        col=col,
        name="Test",
        color=_OBJ_TEST,
    )

    fig.update_scenes(
        xaxis_title="y1 (norm)",
        yaxis_title="y2 (norm)",
        zaxis_title=z_label,
        row=row,
        col=col,
    )


def _plot_3d(
    fig: go.Figure,
    data: dict[str, Any],
    x_key: str,
    y_key: str,
    z_key: str,
    x_col: int,
    y_col: int,
    z_col: int,
    row: int,
    col: int,
    name: str,
    color: str,
):
    xsrc = _get_2d(data, x_key)
    ysrc = _get_2d(data, y_key)
    zsrc = _get_2d(data, z_key)

    if min(xsrc.size, ysrc.size, zsrc.size) > 0:
        add_3d_overlay(
            fig,
            row=row,
            col=col,
            x=xsrc[:, x_col],
            y=ysrc[:, y_col],
            z=zsrc[:, z_col],
            name=name,
            size=3,
            opacity=0.8,
            color=color,
        )


def _get_2d(data: dict[str, Any], key: str):
    arr = np.asarray(data.get(key, []))
    if arr.size == 0:
        return np.empty((0, 2))
    return arr[:, :2] if arr.ndim == 2 else arr.reshape(-1, 2)
