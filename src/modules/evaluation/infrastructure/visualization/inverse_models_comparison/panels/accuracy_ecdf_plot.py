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
        title=dict(
            text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        xaxis=dict(
            title="Best-shot discrepancy (log scale)",
            type="log",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            gridcolor="rgba(211, 211, 211, 0.5)",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
            autorange=True,
        ),
        yaxis=dict(
            title="Fraction of targets",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            gridcolor="rgba(211, 211, 211, 0.5)",
            range=[0, 1.05],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
            dtick=0.2,
        ),
        template="plotly_white",
        height=800,
        width=1000,
        margin=dict(t=100, b=100, l=100, r=100),
        legend=dict(
            font=dict(size=13),
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
    )
    return fig
