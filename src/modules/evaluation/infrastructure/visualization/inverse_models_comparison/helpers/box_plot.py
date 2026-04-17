from typing import List, Optional, Union

import numpy as np
import plotly.graph_objects as go


def add_box_trace(
    fig: go.Figure,
    y: Union[List, np.ndarray],
    name: str,
    color: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
    x: Optional[Union[List, np.ndarray]] = None,
    showlegend: bool = True,
    legendgroup: Optional[str] = None,
    offsetgroup: Optional[str] = None,
    boxpoints: str = "outliers",
    boxmean: Union[bool, str] = True,
    opacity: float = 1.0,
    marker_size: int = 3,
    line_width: float = 1.5,
) -> None:
    """Adds a standardized box plot trace to a figure."""
    trace = go.Box(
        y=y,
        x=x,
        name=name,
        marker=dict(color=color, opacity=opacity, size=marker_size),
        line=dict(width=line_width),
        boxpoints=boxpoints,
        boxmean=boxmean,
        legendgroup=legendgroup,
        showlegend=showlegend,
        offsetgroup=offsetgroup,
    )

    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)
