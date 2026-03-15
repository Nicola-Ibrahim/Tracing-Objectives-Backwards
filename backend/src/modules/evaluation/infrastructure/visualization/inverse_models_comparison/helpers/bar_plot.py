from typing import List, Optional

import plotly.graph_objects as go


def add_bar_trace(
    fig: go.Figure,
    x: List,
    y: List,
    name: str,
    color: str,
    border_color: str,
    row: int = 1,
    col: int = 1,
    showlegend: bool = True,
    legendgroup: Optional[str] = None,
    width: float = 1.5,
) -> None:
    """Adds a standardized bar trace to a figure subplot."""
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=name,
            marker=dict(
                color=color,
                line=dict(color=border_color, width=width),
            ),
            showlegend=showlegend,
            legendgroup=legendgroup,
        ),
        row=row,
        col=col,
    )
