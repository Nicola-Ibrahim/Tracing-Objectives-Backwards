import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.box_plot import add_box_trace


def create_error_boxplot_figure(
    residuals_data: dict[str, list[float]],
    color_map: dict[str, str],
    model_names: list[str],
    title: str,
    subtitle: str,
) -> go.Figure:
    """
    Creates a boxplot figure showing the distribution of residuals.
    """
    fig = make_subplots(rows=1, cols=1)
    row, col = 1, 1

    for model_name in model_names:
        if model_name in residuals_data:
            best_shots = residuals_data[model_name]
            color = color_map.get(model_name, "gray")

            add_box_trace(
                fig=fig,
                y=best_shots,
                name=model_name,
                color=color,
                row=row,
                col=col,
                showlegend=False,
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
            title="Standardized Residual Magnitude",
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
        height=800,
        width=1200,
        margin=dict(t=100, b=100, l=100, r=100),
        showlegend=False,
    )
    return fig
