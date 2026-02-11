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
        title=f"<b>{title}</b><br><sup>{subtitle}</sup>",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Observed Frequency",
        template="plotly_white",
        height=800,
        width=1000,
        showlegend=True,
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
