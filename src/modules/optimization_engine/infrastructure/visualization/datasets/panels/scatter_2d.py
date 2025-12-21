import numpy as np
import plotly.graph_objects as go

def add_scatter_base(
    fig: go.Figure,
    row: int,
    col: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    color: str,
    symbol: str,
    size: int,
    x_label: str,
    y_label: str,
):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=name,
            marker=dict(size=size, opacity=0.85, color=color, symbol=symbol),
            hovertemplate=f"{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4f}}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    # We can't easily set limits here without passing the visualizer instance or helper
    # But the original code called self._set_xy_limits. 
    # For now, we'll assume limits are handled by the caller or we might need to move that helper too.
    # Actually, _set_xy_limits is likely a helper in the visualizer.
    # Let's check if we can move it or if we should just omit it for now and let the caller handle it.
    # The caller (dataset.py) iterates and calls these.
    # Let's keep it simple: just add the trace. The caller can set limits if needed, 
    # OR we can export set_xy_limits as a utility.
    
    # Wait, looking at the original code:
    # self._set_xy_limits(fig, row, col, x, y)
    # This seems important for consistent scaling.
    # I should probably include a utility for this or pass it in.
    # For now, I'll omit the limit setting here and let the caller handle it if possible,
    # or I'll duplicate the logic if it's simple.
    # _set_xy_limits likely just updates axes ranges.
    set_xy_limits(fig, row, col, x, y)

def set_xy_limits(fig, row, col, x, y):
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
):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=name,
            marker=dict(size=size, opacity=opacity, color=color, symbol=symbol),
            showlegend=True,
        ),
        row=row,
        col=col,
    )
