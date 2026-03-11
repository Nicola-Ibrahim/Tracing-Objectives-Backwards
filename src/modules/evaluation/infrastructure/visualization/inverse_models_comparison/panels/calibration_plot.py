from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_plot import add_line_trace


def create_calibration_figure(
    calibration_data: dict[str, dict[str, Any]],
    color_map: dict[str, str],
    title: str,
    subtitle: str,
) -> go.Figure:
    """
    Creates Calibration Curve (PIT) figure using pre-calculated data.
    """

    fig = make_subplots(rows=1, cols=1)
    row, col = 1, 1

    # Add ideal line

    add_line_trace(
        fig=fig,
        x=[0, 1],
        y=[0, 1],
        name="Ideal",
        color="black",
        row=row,
        col=col,
        dash="dash",
        showlegend=True,
    )

    for model_name, data in calibration_data.items():
        color = color_map.get(model_name, "gray")
        pit_values = data["pit_values"]
        cdf_y = data["cdf_y"]

        add_line_trace(
            fig=fig,
            x=pit_values,
            y=cdf_y,
            name=f"{model_name}",
            color=color,
            row=row,
            col=col,
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
            title="Theoretical Quantiles",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            gridcolor="rgba(211, 211, 211, 0.5)",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
            dtick=0.1,
            range=[0, 1],
        ),
        yaxis=dict(
            title="Observed Frequency",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            gridcolor="rgba(211, 211, 211, 0.5)",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
            dtick=0.1,
            range=[0, 1],
        ),
        template="plotly_white",
        height=800,
        width=1000,
        margin=dict(t=100, b=100, l=100, r=100),
        legend=dict(
            font=dict(size=14),
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

    # # Annotations
    # fig.add_annotation(
    #     x=0.8,
    #     y=0.2,
    #     text="Overconfident",
    #     showarrow=False,
    #     font=dict(size=10, color="gray"),
    #     row=row,
    #     col=col,
    # )
    # fig.add_annotation(
    #     x=0.2,
    #     y=0.8,
    #     text="Underconfident",
    #     showarrow=False,
    #     font=dict(size=10, color="gray"),
    #     row=row,
    #     col=col,
    # )
