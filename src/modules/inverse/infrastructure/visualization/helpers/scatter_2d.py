import numpy as np
import plotly.graph_objects as go


def set_xy_limits(fig, row, col, x, y):
    """
    Sets the x and y axis ranges for a given subplot with a 10% padding.
    """
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    xr, yr = x_max - x_min, y_max - y_min
    x_pad = xr * 0.1 if xr > 0 else 0.1
    y_pad = yr * 0.1 if yr > 0 else 0.1
    fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], row=row, col=col)
    fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], row=row, col=col)


def add_scatter_overlay(
    fig: go.Figure,
    row: int,
    col: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    symbol: str,
    size: int,
    opacity: float,
    color: str,
    show_legend: bool = True,
):
    """
    Adds a scatter trace to a specific subplot.
    """
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=name,
            marker=dict(size=size, opacity=opacity, color=color, symbol=symbol),
            showlegend=show_legend,
            hovertemplate=f"{name}: %{{x:.4f}}<br>{name}: %{{y:.4f}}<extra></extra>",
        ),
        row=row,
        col=col,
    )
