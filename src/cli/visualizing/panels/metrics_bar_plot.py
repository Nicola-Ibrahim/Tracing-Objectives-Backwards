from typing import Any

import plotly.graph_objects as go

from ..color_utils import darken_rgba


def add_metrics_bar_plots(
    fig: go.Figure,
    results_map: dict[str, dict[str, Any]],
    color_map: dict[str, str],
    model_names: list[str],
) -> None:
    """
    Adds metrics bar charts (Best Shot, Calibration Error, Diversity) to Row 2.
    Each model is added as a separate trace and linked to the legend via legendgroup.
    """

    # 1. Calibration Residual (Row 1, Col 2)
    for model_name in model_names:
        if model_name in results_map:
            calibration = results_map[model_name]["calibration"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[calibration.get("calibration_error", 0)],
                    marker=dict(
                        color=color,
                        line=dict(color=darken_rgba(color), width=1.5),
                    ),
                    showlegend=False,
                    legendgroup=model_name,
                    name="Calibration Error (Dist to Diagonal)",
                ),
                row=1,
                col=2,
            )

    # 2. CRPS (Row 1, Col 3)
    for model_name in model_names:
        if model_name in results_map:
            calibration = results_map[model_name]["calibration"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[calibration.get("mean_crps", 0)],
                    marker=dict(
                        color=color,
                        line=dict(color=darken_rgba(color), width=1.5),
                    ),
                    showlegend=False,
                    legendgroup=model_name,
                    name="CRPS",
                ),
                row=1,
                col=3,
            )

    # 3. Mean Lowest Residual (Row 2, Col 2)
    for model_name in model_names:
        if model_name in results_map:
            performance = results_map[model_name]["performance"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[performance["mean_lowest_residual"]],
                    marker=dict(
                        color=color,
                        line=dict(color=darken_rgba(color), width=1.5),
                    ),
                    showlegend=False,
                    legendgroup=model_name,
                    name="Mean Lowest Residual",
                ),
                row=2,
                col=2,
            )

    # 4. Mean Reliability (Row 2, Col 3)
    for model_name in model_names:
        if model_name in results_map:
            performance = results_map[model_name]["performance"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[performance.get("mean_reliability", 0)],
                    marker=dict(
                        color=color,
                        line=dict(color=darken_rgba(color), width=1.5),
                    ),
                    showlegend=False,
                    legendgroup=model_name,
                    name="Mean Reliability",
                ),
                row=2,
                col=3,
            )

    # 5. Diversity (Row 3, Col 1)
    for model_name in model_names:
        if model_name in results_map:
            performance = results_map[model_name]["performance"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[performance["mean_diversity"]],
                    marker=dict(
                        color=color,
                        line=dict(color=darken_rgba(color), width=1.5),
                    ),
                    showlegend=False,
                    legendgroup=model_name,
                    name="Diversity",
                ),
                row=3,
                col=1,
            )

    # 6. Interval Width (Row 3, Col 2)
    for model_name in model_names:
        if model_name in results_map:
            performance = results_map[model_name]["performance"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[performance.get("mean_interval_width", 0)],
                    marker=dict(
                        color=color,
                        line=dict(color=darken_rgba(color), width=1.5),
                    ),
                    showlegend=False,
                    legendgroup=model_name,
                    name="Interval Width",
                ),
                row=3,
                col=2,
            )
