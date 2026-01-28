from typing import Any, List, Union

import plotly.graph_objects as go

from ..color_utils import darken_rgba


def add_metric_bar_plot(
    fig: go.Figure,
    results_map: dict[str, dict[str, Any]],
    color_map: dict[str, str],
    model_names: list[str],
    metric_path: List[str],
    metric_name: str,
    row: int = 1,
    col: int = 1,
    show_legend: bool = False,
) -> None:
    """
    Generic helper to add a bar chart for a nested metric.
    Recursively follows metric_path into the result dictionary.
    """

    def _get_nested(data: dict, path: List[str]) -> Union[float, int, None]:
        curr = data
        for key in path:
            if isinstance(curr, dict) and key in curr:
                curr = curr[key]
            else:
                return None
        return curr

    for model_name in model_names:
        if model_name in results_map:
            val = _get_nested(results_map[model_name], metric_path)
            if val is None:
                continue

            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[val],
                    marker=dict(
                        color=color,
                        line=dict(color=darken_rgba(color), width=1.5),
                    ),
                    showlegend=show_legend,
                    legendgroup=model_name,
                    name=model_name if show_legend else metric_name,
                ),
                row=row,
                col=col,
            )
