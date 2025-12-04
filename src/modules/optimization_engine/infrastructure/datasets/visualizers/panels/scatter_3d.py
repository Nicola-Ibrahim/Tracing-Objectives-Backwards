import numpy as np
import plotly.graph_objects as go

def add_3d_overlay(
    fig: go.Figure,
    row: int,
    col: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    name: str,
    size: int,
    opacity: float,
    color: str,
):
    # Align and mask finite rows
    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]

    if x.size == 0:
        return

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name=name,
            marker=dict(size=size, opacity=opacity, color=color),
            hovertemplate="x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
