from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_plot import add_line_trace


def create_accuracy_ecdf_figure(
    ecdf_data: dict[str, dict[str, Any]],
    color_map: dict[str, str],
    title: str,
    subtitle: str,
) -> go.Figure:
    """
    Creates Empirical Cumulative Distribution Function (ECDF) figure.
    """

    fig = make_subplots(rows=1, cols=1)
    row, col = 1, 1

    for model_name, data in ecdf_data.items():
        x_sorted = data["x_sorted"]
        y_ecdf = data["y_ecdf"]
        label = data["label"]
        median_val = data["median_val"]

        color = color_map.get(model_name, "gray")

        # 4. Add ECDF trace
        add_line_trace(
            fig=fig,
            x=x_sorted,
            y=y_ecdf,
            name=label,
            color=color,
            row=row,
            col=col,
            width=2.5,
            shape="hv",
            legendgroup=model_name,
        )

        add_line_trace(
            fig=fig,
            x=[median_val, median_val],
            y=[0, 1],
            name=label,
            color=color,
            row=row,
            col=col,
            width=1,
            dash="dash",
            showlegend=False,
            hoverinfo="skip",
            legendgroup=model_name,
        )

    fig.update_layout(
        title=f"<b>{title}</b><br><sup>{subtitle}</sup>",
        xaxis_title="Best-shot discrepancy (min over K)",
        yaxis_title="Fraction of targets",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        height=800,
        width=1000,
        showlegend=True,
    )
    return fig
