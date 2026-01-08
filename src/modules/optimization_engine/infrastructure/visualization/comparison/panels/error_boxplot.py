from typing import Any

import plotly.graph_objects as go


def add_error_boxplot(
    fig: go.Figure,
    row: int,
    col: int,
    results_map: dict[str, dict[str, Any]],
    color_map: dict[str, str],
    model_names: list[str],
) -> None:
    """
    Adds a boxplot showing the distribution of the lowest residuals achieved
    across all test samples for each model.
    """
    for model_name in model_names:
        if model_name in results_map:
            res = results_map[model_name]
            # Use pre-calculated residuals distributions
            best_shots = res["performance"]["distributions"]["lowest_residual"]

            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Box(
                    y=best_shots,
                    name=model_name,
                    boxmean=True,
                    marker_color=color,
                    showlegend=False,
                    legendgroup=model_name,
                ),
                row=row,
                col=col,
            )
