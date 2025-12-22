from typing import Any

import numpy as np
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
    Adds Re-simulation Error Boxplot to the specified subplot using raw error data.
    """
    for model_name in model_names:
        if model_name in results_map:
            res = results_map[model_name]
            # Use raw errors to get the distribution of best shots across test samples
            raw_errors = res["raw_errors"]
            best_shots = np.min(raw_errors, axis=1)

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
