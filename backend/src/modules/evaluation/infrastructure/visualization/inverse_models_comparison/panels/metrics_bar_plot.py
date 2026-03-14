import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..color_utils import darken_rgba
from ..helpers.bar_plot import add_bar_trace


def create_metric_bar_figure(
    metric_values: dict[str, float],
    color_map: dict[str, str],
    model_names: list[str],
    title: str,
    subtitle: str,
) -> go.Figure:
    """
    Creates a bar chart figure for a specific metric.
    """
    fig = make_subplots(rows=1, cols=1)
    row, col = 1, 1

    for model_name in model_names:
        if model_name in metric_values:
            val = metric_values[model_name]
            color = color_map.get(model_name, "gray")

            add_bar_trace(
                fig=fig,
                x=[model_name],
                y=[val],
                name=model_name,
                color=color,
                border_color=darken_rgba(color),
                row=row,
                col=col,
                showlegend=True,
                legendgroup=model_name,
            )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
            font=dict(size=24),
            x=0.05,
            xanchor="left",
        ),
        yaxis=dict(
            title="Metric Value",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            gridcolor="rgba(211, 211, 211, 0.5)",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
        ),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=False,
        ),
        template="plotly_white",
        height=700,
        width=1200,
        margin=dict(t=120, b=100, l=100, r=100),
        showlegend=False,
    )
    return fig
