import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def add_pdf1d(
    fig: go.Figure,
    row: int,
    col: int,
    v: np.ndarray,
    label: str,
    color: str = "#888",
    kde_color: str | None = None,
):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    fig.add_trace(
        go.Histogram(
            x=v,
            nbinsx=40,
            histnorm="probability density",
            name="Histogram",
            opacity=0.3,
            marker=dict(color=color),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    kde_line_color = kde_color if kde_color else color
    try:
        kde = gaussian_kde(v)
        lo, hi = float(v.min()), float(v.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        grid = np.linspace(lo, hi, 300)
        pdf = kde(grid)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=pdf,
                mode="lines",
                name="KDE",
                line=dict(width=3, color=kde_line_color),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    finally:
        fig.update_xaxes(title_text=label, row=row, col=col)
        fig.update_yaxes(title_text="Density", row=row, col=col)
