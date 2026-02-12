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
        title=dict(
            text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        height=800,
        width=1200,
        boxmode="group",
        template="plotly_white",
        margin=dict(t=100, b=100, l=100, r=100),
        yaxis=dict(
            title="Error Magnitude (L2 Norm)",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            gridcolor="rgba(211, 211, 211, 0.5)",
            zerolinecolor="rgba(211, 211, 211, 0.8)",
            range=[0, 0.2],
            dtick=0.04,
        ),
        xaxis=dict(
            title="Estimator Version",
            title_font=dict(size=18),
            tickfont=dict(size=14),
        ),
        legend=dict(
            title=dict(text="Error Component", font=dict(size=16)),
            font=dict(size=14),
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=False)
    return fig
