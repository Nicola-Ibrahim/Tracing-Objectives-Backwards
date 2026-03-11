from typing import List, Optional, Union

import numpy as np
import plotly.graph_objects as go


def add_line_trace(
    fig: go.Figure,
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    name: str,
    color: str,
    row: int,
    col: int,
    mode: str = "lines",
    dash: Optional[str] = None,
    width: float = 2.0,
    shape: str = "linear",
    showlegend: bool = True,
    legendgroup: Optional[str] = None,
    hoverinfo: str = "all",
) -> None:
    """Adds a standardized line/scatter trace to a figure subplot."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name,
            line=dict(color=color, width=width, dash=dash, shape=shape),
            legendgroup=legendgroup,
            showlegend=showlegend,
            hoverinfo=hoverinfo,
        ),
        row=row,
        col=col,
    )
