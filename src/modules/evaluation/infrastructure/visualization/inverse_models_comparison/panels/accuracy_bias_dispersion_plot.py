from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.box_plot import add_box_trace


def create_accuracy_bias_dispersion_figure(
    bias_data: dict[str, list[float]],
    dispersion_data: dict[str, list[float]],
    model_names: List[str],
    title: str,
    subtitle: str,
) -> go.Figure:
    """
    Creates a grouped box plot figure for Bias and Dispersion.
    """
    fig = make_subplots(rows=1, cols=1)

    # 1. Colors
    COLOR_BIAS = "#003f5c"  # Deep Navy Blue
    COLOR_DISPERSION = "#488a5c"  # Muted Sage Green

    # 2. Add traces for each estimator

    for model_name in model_names:
        b = bias_data.get(model_name, [])
        v = dispersion_data.get(model_name, [])

        if len(b) == 0 or len(v) == 0:
            continue

        # Add Bias Trace
        add_box_trace(
            fig=fig,
            y=b,
            x=[model_name] * len(b),
            name="Bias (Systematic Error)",
            color=COLOR_BIAS,
            opacity=0.3,
            offsetgroup="Bias",
            legendgroup="Bias",
            showlegend=(model_name == model_names[0]),
        )

        # Add Dispersion Trace
        add_box_trace(
            fig=fig,
            y=v,
            x=[model_name] * len(v),
            name="Dispersion (Random Error)",
            color=COLOR_DISPERSION,
            opacity=0.3,
            offsetgroup="Dispersion",
            legendgroup="Dispersion",
            showlegend=(model_name == model_names[0]),
        )

    fig.update_layout(
        title=f"<b>{title}</b><br><sup>{subtitle}</sup>",
        height=800,
        width=1200,
        boxmode="group",
        template="plotly_white",
        yaxis=dict(
            title="Error Magnitude (L2 Norm)",
            gridcolor="lightgrey",
            zerolinecolor="lightgrey",
            range=[0, 0.2],
        ),
        xaxis=dict(
            title="Estimator Version",
        ),
        legend=dict(
            title="Error Component",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="lightgrey", mirror=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="lightgrey", mirror=False)
    return fig
